"""
Swadesh v4 Training Core
mask function, eval function, training loop

V4 change: desc from13→26 tokens (13dim × 2 tokens/dim)。
mask and eval per dimension, 2 tokens per dim masked/evaluated together.

This file contains no phase definitions or data configs.
change phase design → edit train_phases.py
edit mask/eval logic → edit here
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

N_DIMS = 13
TOKENS_PER_DIM = 2   # V4: [degree_mark, base_word]
DESC_SLOTS = N_DIMS * TOKENS_PER_DIM  # 26

# ═══════════════════════════════════════════════════════
# Data preprocessing
# ═══════════════════════════════════════════════════════

def pretensorize(seqs, pad_id):
    """list[list[int]] → padded tensor (CPU)"""
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long)
    mx = max(len(s) for s in seqs)
    arr = np.full((len(seqs), mx), pad_id, dtype=np.int64)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    return torch.from_numpy(arr)


def repeat_tensor(t, n):
    """Repeat tensor n times"""
    if t.shape[0] == 0 or n <= 1:
        return t
    return t.repeat(n, 1)


def repeat_eidx(e, n):
    if e is None:
        return None
    return e.repeat(n)


# ═══════════════════════════════════════════════════════
# Mask functions
# ═══════════════════════════════════════════════════════

def mask_p_region(batch, mr, pad_id, mask_id, p_start=2, p_len=13):
    """
    S1 mask：random mask in P region。
    Modelinfer full perception from partial。
    """
    B, L = batch.shape
    masked = batch.clone()
    targets = torch.full_like(batch, -100)

    p_end = min(p_start + p_len, L)
    if p_end <= p_start:
        return masked, targets

    region = batch[:, p_start:p_end]
    rand = torch.rand(B, p_end - p_start, device=batch.device)
    not_pad = (region != pad_id)
    to_mask = not_pad & (rand < mr)

    neg = torch.tensor(-100, dtype=batch.dtype, device=batch.device)
    for d in range(p_end - p_start):
        pos = p_start + d
        masked[:, pos] = torch.where(to_mask[:, d], mask_id, batch[:, pos])
        targets[:, pos] = torch.where(to_mask[:, d], batch[:, pos], neg)

    return masked, targets


def mask_p_in_s2(batch, mr, pad_id, mask_id, p_start, p_len=13):
    """
    S2 Reverse mask：Mask P region in S2 sequence, keep desc。
    Model infers perceptual encoding (P) from language description (desc)。
    
    Same as mask_p_region same logic，but for S2 format。
    p_start = desc_start - 13。
    """
    B, L = batch.shape
    masked = batch.clone()
    targets = torch.full_like(batch, -100)

    p_end = min(p_start + p_len, L)
    if p_end <= p_start:
        return masked, targets

    region = batch[:, p_start:p_end]
    rand = torch.rand(B, p_end - p_start, device=batch.device)
    not_pad = (region != pad_id)
    to_mask = not_pad & (rand < mr)

    neg = torch.tensor(-100, dtype=batch.dtype, device=batch.device)
    for d in range(p_end - p_start):
        pos = p_start + d
        masked[:, pos] = torch.where(to_mask[:, d], mask_id, batch[:, pos])
        targets[:, pos] = torch.where(to_mask[:, d], batch[:, pos], neg)

    return masked, targets


def mask_desc_region(batch, mr, pad_id, mask_id, desc_start,
                     n_dims=N_DIMS, tpd=TOKENS_PER_DIM,
                     eidx=None, holdout_dims=None):
    """
    V4: Mask desc region by dimension。tpd tokens per dimension masked together。
    n_dims=13, tpd=2 → descregion26tokens。
    """
    B, L = batch.shape
    masked = batch.clone()
    targets = torch.full_like(batch, -100)
    neg = torch.tensor(-100, dtype=batch.dtype, device=batch.device)

    for d in range(n_dims):
        pos0 = desc_start + d * tpd       # degree mark
        pos1 = desc_start + d * tpd + 1   # base word
        if pos1 >= L:
            break

        rand = torch.rand(B, device=batch.device)
        not_pad = (batch[:, pos0] != pad_id)
        to_mask = not_pad & (rand < mr)

        # Exclude held-out dimensions
        if eidx is not None and holdout_dims is not None:
            for b in range(B):
                ei = int(eidx[b])
                if 0 <= ei < len(holdout_dims):
                    if d in holdout_dims[ei]:
                        to_mask[b] = False

        masked[:, pos0] = torch.where(to_mask, mask_id, batch[:, pos0])
        masked[:, pos1] = torch.where(to_mask, mask_id, batch[:, pos1])
        targets[:, pos0] = torch.where(to_mask, batch[:, pos0], neg)
        targets[:, pos1] = torch.where(to_mask, batch[:, pos1], neg)

    return masked, targets


# ═══════════════════════════════════════════════════════
# Evaluation functions
# ═══════════════════════════════════════════════════════

def eval_p_recon(model, tensor, device, pad_id, mask_id,
                 p_start=2, p_len=13, eval_bs=128):
    """S1：Mask each P position, test reconstruction。Return (overall, per_dim_dict)"""
    if tensor.shape[0] == 0:
        return 0.0, {}
    B, L = tensor.shape
    dim_names = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    per_dim_ok = [0] * p_len
    per_dim_total = [0] * p_len
    model.eval()

    with torch.no_grad():
        for d in range(p_len):
            pos = p_start + d
            if pos >= L:
                break
            inp = tensor.clone()
            gold = tensor[:, pos].clone()
            valid = (gold != pad_id)
            if valid.sum() == 0:
                continue
            inp[:, pos] = mask_id
            for i in range(0, B, eval_bs):
                j = min(i + eval_bs, B)
                batch = inp[i:j].to(device)
                logits = model(batch)
                preds = logits[:, pos].argmax(dim=-1).cpu()
                v = valid[i:j]
                per_dim_ok[d] += ((preds == gold[i:j]) & v).sum().item()
                per_dim_total[d] += v.sum().item()

    per_dim = {}
    total_ok, total_n = 0, 0
    for d in range(p_len):
        name = dim_names[d] if d < len(dim_names) else f'D{d}'
        if per_dim_total[d] > 0:
            per_dim[name] = round(per_dim_ok[d] / per_dim_total[d], 3)
            total_ok += per_dim_ok[d]
            total_n += per_dim_total[d]
        else:
            per_dim[name] = 0.0
    overall = total_ok / total_n if total_n > 0 else 0.0
    return overall, per_dim


def eval_desc_acc(model, tensor, device, pad_id, mask_id, desc_start,
                  n_dims=N_DIMS, tpd=TOKENS_PER_DIM,
                  eval_bs=128, target_dims=None):
    """V4: Mask each dimension separately（per dimtpdtokens），Both correct counts as dimension correct。"""
    if tensor.shape[0] == 0:
        return 0.0, {}
    B, L = tensor.shape
    dim_names = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    if target_dims is None:
        target_dims = list(range(n_dims))

    per_dim_ok = {d: 0 for d in target_dims}
    per_dim_total = {d: 0 for d in target_dims}

    model.eval()
    with torch.no_grad():
        for d in target_dims:
            pos0 = desc_start + d * tpd
            pos1 = desc_start + d * tpd + 1
            if pos1 >= L:
                continue
            inp = tensor.clone()
            gold0 = tensor[:, pos0].clone()
            gold1 = tensor[:, pos1].clone()
            valid = (gold0 != pad_id) & (gold1 != pad_id)
            if valid.sum() == 0:
                continue
            inp[:, pos0] = mask_id
            inp[:, pos1] = mask_id
            for i in range(0, B, eval_bs):
                j = min(i + eval_bs, B)
                batch = inp[i:j].to(device)
                logits = model(batch)
                pred0 = logits[:, pos0].argmax(dim=-1).cpu()
                pred1 = logits[:, pos1].argmax(dim=-1).cpu()
                v = valid[i:j]
                both = (pred0 == gold0[i:j]) & (pred1 == gold1[i:j]) & v
                per_dim_ok[d] += both.sum().item()
                per_dim_total[d] += v.sum().item()

    per_dim = {}
    total_ok, total_n = 0, 0
    for d in target_dims:
        name = dim_names[d] if d < len(dim_names) else f'D{d}'
        if per_dim_total[d] > 0:
            per_dim[name] = round(per_dim_ok[d] / per_dim_total[d], 3)
            total_ok += per_dim_ok[d]
            total_n += per_dim_total[d]
        else:
            per_dim[name] = 0.0
    return total_ok / total_n if total_n > 0 else 0.0, per_dim


def eval_desc_holdout(model, tensor, device, pad_id, mask_id, desc_start,
                      eidx_list, holdout_dims, n_dims=N_DIMS,
                      tpd=TOKENS_PER_DIM, eval_bs=128):
    """V4: held-outdimensionsevaluation。per dimtpdtokenssimultaneouslymask，both correct to count。"""
    if tensor.shape[0] == 0:
        return 0.0
    B, L = tensor.shape
    model.eval()
    ok, total = 0, 0

    with torch.no_grad():
        for d in range(n_dims):
            pos0 = desc_start + d * tpd
            pos1 = desc_start + d * tpd + 1
            if pos1 >= L:
                continue
            is_held = torch.zeros(B, dtype=torch.bool)
            for b in range(B):
                ei = eidx_list[b]
                if ei >= 0 and d in holdout_dims[ei]:
                    is_held[b] = True
            if is_held.sum() == 0:
                continue
            idx = torch.where(is_held)[0]
            sub = tensor[idx].clone()
            gold0 = tensor[idx, pos0].clone()
            gold1 = tensor[idx, pos1].clone()
            sub[:, pos0] = mask_id
            sub[:, pos1] = mask_id
            for i in range(0, len(idx), eval_bs):
                j = min(i + eval_bs, len(idx))
                batch = sub[i:j].to(device)
                logits = model(batch)
                pred0 = logits[:, pos0].argmax(dim=-1).cpu()
                pred1 = logits[:, pos1].argmax(dim=-1).cpu()
                both = (pred0 == gold0[i:j]) & (pred1 == gold1[i:j])
                ok += both.sum().item()
                total += (j - i)
    return ok / total if total > 0 else 0.0


def eval_full_mask(model, tensor, device, pad_id, mask_id, desc_start,
                   n_dims=N_DIMS, tpd=TOKENS_PER_DIM, eval_bs=128):
    """V4: MASK all desc positions simultaneously。per dimtpdtokens，Both correct counts as dimension correct。"""
    if tensor.shape[0] == 0:
        return 0.0, {}
    B, L = tensor.shape
    dim_names = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    per_dim_ok = [0] * n_dims
    per_dim_total = [0] * n_dims

    inp = tensor.clone()
    for d in range(n_dims):
        pos0 = desc_start + d * tpd
        pos1 = desc_start + d * tpd + 1
        if pos1 < L:
            inp[:, pos0] = mask_id
            inp[:, pos1] = mask_id

    model.eval()
    with torch.no_grad():
        for i in range(0, B, eval_bs):
            j = min(i + eval_bs, B)
            batch = inp[i:j].to(device)
            logits = model(batch)
            for d in range(n_dims):
                pos0 = desc_start + d * tpd
                pos1 = desc_start + d * tpd + 1
                if pos1 >= L:
                    continue
                gold0 = tensor[i:j, pos0]
                gold1 = tensor[i:j, pos1]
                valid = (gold0 != pad_id) & (gold1 != pad_id)
                pred0 = logits[:, pos0].argmax(dim=-1).cpu()
                pred1 = logits[:, pos1].argmax(dim=-1).cpu()
                both = (pred0 == gold0) & (pred1 == gold1) & valid
                per_dim_ok[d] += both.sum().item()
                per_dim_total[d] += valid.sum().item()

    per_dim = {}
    total_ok, total_n = 0, 0
    for d in range(n_dims):
        name = dim_names[d] if d < len(dim_names) else f'D{d}'
        if per_dim_total[d] > 0:
            per_dim[name] = round(per_dim_ok[d] / per_dim_total[d], 3)
            total_ok += per_dim_ok[d]
            total_n += per_dim_total[d]
        else:
            per_dim[name] = 0.0
    return total_ok / total_n if total_n > 0 else 0.0, per_dim


def per_dimension_probe(model, entities, device, t2id, n_dims=13):
    """LOO linear probe on entity tokens embeddings"""
    model.eval()
    names_enc = [(n, e) for n, e in entities if n in t2id]
    if not names_enc:
        return {}
    ids = torch.tensor([t2id[n] for n, _ in names_enc], device=device)
    with torch.no_grad():
        X = model.tokens_embed(ids).cpu().numpy()
    Y = {d: [e[d] for _, e in names_enc] for d in range(n_dims)}
    dim_names = ['T','H','M','V','Z','A','W','R','O','S','F','L','An']
    results = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        import warnings; warnings.filterwarnings('ignore')
        for d in range(n_dims):
            y = np.array(Y[d])
            if len(set(y)) < 2:
                results[dim_names[d]] = 0.0; continue
            clf = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
            scores = cross_val_score(clf, X, y, cv=LeaveOneOut())
            results[dim_names[d]] = round(scores.mean(), 3)
    except ImportError:
        for d in range(n_dims):
            results[dim_names[d]] = -1.0
    return results


# ═══════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════

def build_tagged_batches(stage_data, bs, rng):
    """
    stage_data: list of (fmt_tag, tensor, eidx_tensor_or_None)
    Return: shuffled list of (fmt_tag, batch_tensor, eidx_batch_or_None)
    """
    all_batches = []
    for fmt_tag, tensor, eidx in stage_data:
        if tensor.shape[0] == 0:
            continue
        n = tensor.shape[0]
        perm = rng.permutation(n)
        tensor = tensor[perm]
        if eidx is not None:
            eidx = eidx[perm]
        for i in range(0, n, bs):
            j = min(i + bs, n)
            eb = eidx[i:j] if eidx is not None else None
            all_batches.append((fmt_tag, tensor[i:j], eb))
    perm = rng.permutation(len(all_batches))
    return [all_batches[p] for p in perm]


def train_epoch(model, opt, tagged_batches, device, pad_id, mask_id, mr,
                holdout_dims=None, use_fp16=False, reverse_ratio=0.0):
    """
    One epoch。tagged_batches from build_tagged_batches。
    
    reverse_ratio: S2format(withParray)in batch，what ratio to doreversemask(mask P, predict from desc)。
                   0.0 = forward-only(original behavior)。0.3 = 30%reverse。
    """
    from v6_tool import FMT_META

    amp_device = device.split(':')[0] if ':' in device else device
    amp_ctx = torch.autocast(device_type=amp_device,
                             dtype=torch.float16, enabled=use_fp16)

    model.train()
    total_loss, total_tokens, steps = 0.0, 0, 0

    for fmt_tag, batch, eidx in tagged_batches:
        batch = batch.to(device)

        if fmt_tag == 's1_bare':
            masked, targets = mask_p_region(batch, mr, pad_id, mask_id,
                                            p_start=2, p_len=13)
        else:
            meta = FMT_META.get(fmt_tag)
            if meta is None:
                continue
            _, desc_start, has_p = meta
            if desc_start is None:
                continue

            # reversemask：S2formatwithParray，withreverse_ratioprobability to mask P instead of desc
            do_reverse = (has_p and reverse_ratio > 0
                          and torch.rand(1).item() < reverse_ratio)

            if do_reverse:
                p_start = desc_start - 13
                masked, targets = mask_p_in_s2(
                    batch, mr, pad_id, mask_id, p_start=p_start)
            else:
                masked, targets = mask_desc_region(
                    batch, mr, pad_id, mask_id, desc_start,
                    eidx=eidx, holdout_dims=holdout_dims)

        with amp_ctx:
            logits = model(masked)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        n_tok = (targets != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        steps += 1

    avg_loss = total_loss / max(total_tokens, 1)
    return steps, avg_loss
