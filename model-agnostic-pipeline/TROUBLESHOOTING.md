# 🔧 Troubleshooting Guide - Model-Agnostic Pipeline

**Critical issues identified and resolved during blockchain verification testing**

## 🚨 **Major Blockers and Solutions**

### **Blocker 1: Missing Hardhat Setup** 
**Error**: `HH700: Artifact for contract "X" not found`

**Root Cause**: MAP generates deployment scripts but has no Hardhat configuration to compile contracts.

**Solution**:
```bash
# One-time setup
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-ethers ethers

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

mkdir -p contracts
```

---

### **Blocker 2: Contract Compilation Missing**
**Error**: Scripts run but can't find compiled artifacts

**Root Cause**: Generated verifier contracts are not copied to contracts/ directory for compilation.

**Solution**:
```bash
# After generating EZKL proof
cp output_<model>/<model>Verifier.sol contracts/
npx hardhat compile
```

---

### **Blocker 3: Incorrect Verification Script Template**
**Error**: Low-level call failures during verification

**Root Cause**: Original verification script used low-level transaction calls instead of contract method calls.

**Solution**: Fixed in demo.py - verification script now uses:
```javascript
const verifier = await hre.ethers.getContractAt("contracts/{model}Verifier.sol:Halo2Verifier", address);
const tx = await verifier.verifyProof(calldata);
```

---

### **Blocker 4: Missing Network Configuration**
**Error**: `Error: could not detect network`

**Root Cause**: Hardhat not configured with localhost network settings.

**Solution**: Added network configuration to hardhat.config.js with unlimited contract size support.

---

## 📋 **Complete Verification Workflow**

**Follow this exact sequence to avoid all blockers:**

### Step 1: One-Time Setup
```bash
# Install dependencies (only needed once)
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-ethers ethers

# Create hardhat.config.js (see SETUP.md)
mkdir -p contracts
```

### Step 2: Generate Proof
```bash
python demo.py --model simple_mlp --blockchain-scripts
```

### Step 3: Compile Contract
```bash
cp output_simple_mlp/simple_mlpVerifier.sol contracts/
npx hardhat compile
```

### Step 4: Start Blockchain
```bash
# In separate terminal
npx hardhat node --port 8545
```

### Step 5: Deploy and Verify
```bash
npx hardhat run output_simple_mlp/deploy_simple_mlp.js --network localhost
npx hardhat run output_simple_mlp/verify_simple_mlp.js --network localhost
```

## ✅ **Verification Success Indicators**

**When working correctly, you should see:**

### Deployment Success:
```
🚀 Deploying simple_mlp Verifier Contract...
⏳ Deploying contract...
✅ simple_mlpVerifier deployed to: 0x[address]
📁 Deployment info saved
```

### Verification Success:
```
🧪 Testing simple_mlp Blockchain Verification...
📍 Verifier contract: 0x[address]
📦 Calldata length: 6858
⛽ Gas estimate: 537000
📋 Transaction hash: 0x[hash]
⛽ Gas used: 537000
✅ Status: SUCCESS
🎉 simple_mlp proof verified on blockchain!
```

## 🐛 **Common Issues**

### Issue: "Multiple artifacts for contract Halo2Verifier"
**Cause**: Multiple verifier contracts compiled
**Fix**: Use fully qualified names in getContractFactory calls (already fixed in templates)

### Issue: "Network connection failed"
**Cause**: Hardhat node not running
**Fix**: Start `npx hardhat node --port 8545` in separate terminal

### Issue: "Contract size exceeds limit"
**Cause**: Large verifier contracts
**Fix**: Added `allowUnlimitedContractSize: true` to network config

### Issue: "Verification reverts without reason"
**Cause**: Wrong calldata format or corrupted proof
**Fix**: Regenerate proof with `python demo.py --model X --blockchain-scripts`

## 📊 **Verified Model Status**

**All models have been proven with actual blockchain verification:**

| Model | Gas Used | Transaction Hash (example) | Status |
|-------|----------|----------------------------|---------|
| simple_mlp | 537K | 0x2a19cd0e... | ✅ PROVEN |
| rwkv_simple | 614K | 0x2a2b71b0... | ✅ PROVEN |
| mamba_simple | 615K | 0x6934963... | ✅ PROVEN |
| xlstm_simple | 616K | 0xcc70b9d2... | ✅ PROVEN |
| weather_xlstm | 800K | (previously verified) | ✅ PROVEN |

## 🎯 **What Was Fixed**

1. **Added complete Hardhat setup instructions** to SETUP.md
2. **Fixed verification script template** in demo.py to use proper contract calls
3. **Updated all model configurations** to reflect PROVEN status with actual gas costs
4. **Added contract compilation steps** to all documentation
5. **Created comprehensive troubleshooting guide** to prevent future issues

## 🔄 **Testing New Models**

**To verify any new model works correctly:**

1. Follow complete setup workflow above
2. Generate proof: `python demo.py --model new_model --blockchain-scripts`
3. Compile and deploy following steps 3-5
4. Confirm you see success indicators
5. Update model config with `proven_working: True` and actual gas cost

This ensures any new models added to the system will have proper blockchain verification support from day one.