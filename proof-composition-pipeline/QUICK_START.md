# ⚡ Quick Start Guide - Proof Composition Pipeline

A fast-track guide to get the sequential proof composition pipeline running in under 10 minutes.

## 🚀 One-Command Setup

```bash
# 1. Start blockchain (keep running)
npx hardhat node --port 8545 &

# 2. Generate, deploy, and test in one go
python demo_sequential_composition.py --chain-type simple && \
cp sequential_composition_output_simple_sequential_processing_chain/*.sol contracts/ && \
npx hardhat compile && \
cd sequential_composition_output_simple_sequential_processing_chain && \
npx hardhat run deploy_sequential_chain.js --network localhost && \
npx hardhat run test_sequential_chain_working.js --network localhost
```

## 📋 Step-by-Step (5 commands)

```bash
# 1. Generate sequential composition
python demo_sequential_composition.py --chain-type simple

# 2. Copy contracts and compile
cp sequential_composition_output_simple_sequential_processing_chain/*.sol contracts/
npx hardhat compile

# 3. Deploy to blockchain
cd sequential_composition_output_simple_sequential_processing_chain
npx hardhat run deploy_sequential_chain.js --network localhost

# 4. Test sequential execution
npx hardhat run test_sequential_chain_working.js --network localhost

# 5. Check results
echo "✅ Sequential proof composition working!"
```

## 🎯 Expected Results

### ✅ Success Indicators
- **Proof Generation**: ~5-10 minutes, generates 3 ZK proofs
- **Contract Compilation**: "Compiled 7 Solidity files successfully"
- **Deployment**: 7 contracts deployed with addresses
- **Test Execution**: "✅ SUCCESS" with gas usage ~2M
- **Final Result**: "Overall success: ✅"

### 📊 Key Metrics
```
⛓️  Sequential Executor: 0x[address]
📊 Enhanced Verifiers: 3 deployed
⛽ Total Gas Used: ~2,053,959
🔗 Transaction Status: SUCCESS
📈 Steps Completed: 3/3 ✅
```

## 🔧 Quick Customization

### Change Chain Type
```bash
# Simple 3-model chain (default)
python demo_sequential_composition.py --chain-type simple

# Weather processing chain
python demo_sequential_composition.py --chain-type weather

# Complex 5-model chain
python demo_sequential_composition.py --chain-type complex
```

### Custom Output Directory
```bash
python demo_sequential_composition.py --chain-type simple --output-dir my_custom_chain
```

## 🐛 Quick Fixes

### If Generation Fails
```bash
# Check EZKL installation
ezkl --version
pip install --upgrade ezkl torch

# Try again
python demo_sequential_composition.py --chain-type simple
```

### If Deployment Fails
```bash
# Restart blockchain
pkill -f "hardhat node"
npx hardhat node --port 8545 &

# Clean and recompile
npx hardhat clean
npx hardhat compile
```

### If Tests Fail
```bash
# Check deployment file exists
ls sequential_composition_output_*/sequential_deployment.json

# Check contracts deployed
npx hardhat console --network localhost
> await ethers.provider.getCode("0x[executor_address]")
```

## 📁 Generated Files Overview

```
sequential_composition_output_simple_sequential_processing_chain/
├── 🏗️  SequentialChainExecutor.sol           # Main orchestrator
├── 🔧 Enhanced*Verifier.sol                   # Wrapper verifiers (3)
├── ⚡ step*Verifier.sol                       # EZKL verifiers (3)
├── 🚀 deploy_sequential_chain.js              # Deployment script
├── 🧪 test_sequential_chain_working.js        # Test script
├── 📋 README.md                               # Documentation
├── 📊 sequential_composition.json             # Metadata
├── 🔗 sequential_deployment.json              # Addresses
└── 📦 step*_calldata.bytes                    # Proof data (3)
```

## 🎯 What You Get

### ✅ Working Sequential Chain
- **3 Models**: feature_extractor → classifier → decision_maker
- **Real ZK Proofs**: EZKL-generated, cryptographically sound
- **On-Chain Execution**: True sequential A→B→C verification
- **Transparent Flow**: All intermediate values visible
- **Gas Efficient**: ~2M gas for complete 3-step verification

### 🔗 Ready for Production
- **Modular Architecture**: Easy to swap models
- **Layer 2 Ready**: Deploy on Polygon, Arbitrum, etc.
- **Debuggable**: Clear failure points and intermediate values
- **Composable**: Add/remove steps without breaking chain

---

## 🚀 What's Next?

1. **Replace Demo Models**: Plug in your trained models
2. **Use Real Data**: Replace random test inputs with actual data
3. **Scale Up**: Add more steps or deploy on Layer 2
4. **Advanced Patterns**: Try parallel, conditional, or tree compositions

**You now have a complete ZK proof composition system running end-to-end! 🎉**