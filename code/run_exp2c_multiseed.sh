#!/bin/bash
# Batch run exp2c dimension ablation · 5 seeds · bidir30
# Usage: cd to project root, then bash run_exp2c_multiseed.sh

BASE="results/bidir_comparison/bidir_30pct"
SEEDS=(42 137 271 314 577)
WORLD="swadesh_v6_world.md"

for s in "${SEEDS[@]}"; do
    DIR="${BASE}/results_v4_ph5_bidir30_s${s}"
    MODEL="${DIR}/model.pt"
    
    if [ ! -f "$MODEL" ]; then
        echo "⚠️ Not found: $MODEL, skipping"
        continue
    fi
    
    echo "════════════════════════════════════════"
    echo "seed ${s}: ${MODEL}"
    echo "════════════════════════════════════════"
    python3 exp2c_dim_ablation.py --model "$MODEL" --world "$WORLD"
    echo ""
done

echo "All done. To aggregate:"
echo "  python3 aggregate_exp2c.py --base ${BASE} --seeds 42,137,271,314,577"
