#!/bin/bash
# Verification Script for Proof-Chained AI Analysis
echo "🔐 Verifying Proof-Chained AI Analysis..."
echo "Pipeline: RWKV → Mamba → xLSTM"
echo ""

cd ezkl_workspace

for model in rwkv_simple mamba_simple xlstm_simple; do
    echo "Verifying $model proof..."
    cd $model
    if ezkl verify; then
        echo "✅ $model proof verified"
    else
        echo "❌ $model proof verification failed"
    fi
    cd ..
    echo ""
done

echo "🎉 Verification complete!"
