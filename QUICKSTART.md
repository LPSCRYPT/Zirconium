# 🚀 Quick Start Guide

**Get Zirconium running in 10 minutes**

## Prerequisites

- **Node.js** (v18+)
- **Python** (3.9+) 
- **EZKL** installed (`cargo install ezkl`)
- **Git**

## 1. Clone and Setup

```bash
git clone <repository-url>
cd zirconium

# Install dependencies
npm install
pip install -r requirements.txt

# Verify EZKL installation
ezkl --version
```

## 2. Run Local Blockchain

```bash
# Terminal 1: Start local hardhat node
npx hardhat node
```

## 3. Deploy Contracts

```bash
# Terminal 2: Deploy to localhost
npx hardhat run scripts/deploy_minimal.js --network localhost
```

## 4. Run Proof Chain Demo

```bash
# Generate and verify 3-model proof chain
python src/blockchain_proof_chain_submission.py
```

**Expected Output:**
```
🚀 End-to-End Proof-Chain Demo
✅ Generated 3 proofs successfully
✅ Blockchain submission successful!
⛽ Gas used: ~1,200,000
```

## 5. Verify Offchain (Optional)

```bash
# Verify same proofs offchain
cd ezkl_workspace/rwkv_simple
ezkl verify --proof-path=proof.json --vk-path=vk.key --settings-path=settings.json
```

## What Just Happened?

1. **Generated real ZK proofs** for RWKV, Mamba, and xLSTM neural networks
2. **Chained the computations** (RWKV output → Mamba input → xLSTM input)
3. **Verified cryptographically** on local blockchain using ~1.2M gas
4. **Proved computational integrity** without revealing model weights

## Next Steps

- See `examples/` for more demos
- Check `docs/APPLICATIONS.md` for use cases
- Read `setup/` guides for testnet deployment