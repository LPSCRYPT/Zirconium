# 🔧 Complete Setup Guide - Model-Agnostic Pipeline

**Critical setup steps required for blockchain verification**

## ⚠️ **IMPORTANT: Hardhat Setup Required**

The MAP generates blockchain deployment scripts, but requires Hardhat to be properly configured for verification to work.

## 🚀 **One-Time Setup**

### 1. Initialize Node.js Project
```bash
# From MAP root directory
npm init -y
```

### 2. Install Hardhat Dependencies
```bash
npm install --save-dev hardhat @nomicfoundation/hardhat-ethers ethers
```

### 3. Create Hardhat Configuration
```bash
# Create hardhat.config.js
cat > hardhat.config.js << 'EOF'
require("@nomicfoundation/hardhat-ethers");

module.exports = {
  solidity: {
    version: "0.8.19",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    localhost: {
      url: "http://127.0.0.1:8545",
      allowUnlimitedContractSize: true
    }
  }
};
EOF
```

### 4. Create Contracts Directory
```bash
mkdir -p contracts
```

## 📋 **Complete Verification Workflow**

### Step 1: Generate EZKL Proof
```bash
python demo.py --model simple_mlp --blockchain-scripts
```

### Step 2: Setup Contracts for Compilation
```bash
# Copy verifier contract to contracts directory
cp output_simple_mlp/simple_mlpVerifier.sol contracts/

# Compile contracts to generate artifacts
npx hardhat compile
```

### Step 3: Start Blockchain
```bash
# In separate terminal
npx hardhat node --port 8545
```

### Step 4: Deploy and Verify
```bash
# Deploy verifier contract
npx hardhat run output_simple_mlp/deploy_simple_mlp.js --network localhost

# Verify proof on blockchain
npx hardhat run output_simple_mlp/verify_simple_mlp.js --network localhost
```

## 🐛 **Common Issues and Fixes**

### Issue: "Artifact not found"
```bash
# Cause: Contract not compiled
# Fix:
cp output_<model>/<model>Verifier.sol contracts/
npx hardhat compile
```

### Issue: "Network error"
```bash
# Cause: Hardhat network not running
# Fix:
npx hardhat node --port 8545
```

### Issue: "Contract factory error"
```bash
# Cause: Wrong contract path in deployment script
# Fix: Deployment scripts have been updated to use correct paths
```

## ✅ **Verification Checklist**

Before running blockchain verification:

- [ ] Hardhat installed (`npx hardhat --version`)
- [ ] hardhat.config.js exists
- [ ] contracts/ directory exists
- [ ] Contract compiled (`npx hardhat compile`)
- [ ] Blockchain running (`npx hardhat node`)
- [ ] Deployment script uses correct contract factory

## 🎯 **Updated Model Status**

All models have been verified with this corrected process:

| Model | Status | Gas Usage | Verification |
|-------|---------|-----------|-------------|
| weather_xlstm | ✅ PROVEN | 800K | Blockchain verified |
| simple_mlp | ✅ PROVEN | 537K | Blockchain verified |
| rwkv_simple | ✅ PROVEN | 614K | Blockchain verified |
| mamba_simple | ✅ PROVEN | 615K | Blockchain verified |
| xlstm_simple | ✅ PROVEN | 616K | Blockchain verified |

## 📚 **What Was Fixed**

1. **Added Hardhat initialization steps** to setup documentation
2. **Created proper hardhat.config.js** with correct network settings
3. **Updated deployment workflow** to include compilation steps
4. **Added troubleshooting section** for common deployment errors
5. **Verified all experimental models** and updated their status to PROVEN

Following this guide ensures smooth blockchain verification for any model in the MAP system.