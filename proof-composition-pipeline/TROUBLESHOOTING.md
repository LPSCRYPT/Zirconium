# 🔧 Troubleshooting Guide - Proof Composition Pipeline

Common issues and solutions when running the sequential proof composition pipeline.

## 🚨 Common Error Categories

### 1. 📦 EZKL/Proof Generation Errors
### 2. 🏗️ Contract Compilation Errors  
### 3. 🚀 Deployment Errors
### 4. ⛓️ Sequential Execution Errors
### 5. ⛽ Gas and Performance Issues

---

## 📦 EZKL/Proof Generation Errors

### ❌ `ezkl: command not found`
```bash
# Problem: EZKL not installed or not in PATH
# Solution:
pip install ezkl
# OR
cargo install ezkl
ezkl --version  # Should show version number
```

### ❌ `ModuleNotFoundError: No module named 'torch'`
```bash
# Problem: PyTorch not installed
# Solution:
pip install torch torchvision
# OR for specific CUDA version:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### ❌ `EZKL circuit compilation failed`
```bash
# Problem: Model too complex or incompatible operations
# Solutions:
1. Simplify your model (remove unsupported operations)
2. Check EZKL compatibility:
   python -c "import torch; print(torch.nn.functional.relu(torch.tensor([1.0])))"
3. Use smaller input sizes
4. Remove batch normalization, dropout, or other unsupported layers
```

### ❌ `Proof generation timeout`
```bash
# Problem: Model too large for proof generation
# Solutions:
1. Reduce model size
2. Use smaller input dimensions
3. Increase timeout in pipeline settings
4. Use faster hardware (more RAM/CPU)
```

---

## 🏗️ Contract Compilation Errors

### ❌ `HH700: Artifact for contract "X" not found`
```bash
# Problem: Contract name mismatch or compilation failed
# Solutions:
npx hardhat clean
npx hardhat compile

# Check contract names:
ls contracts/Enhanced*
ls artifacts/contracts/
```

### ❌ `ParserError: Expected identifier`
```bash
# Problem: Solidity syntax error in generated contracts
# Solutions:
1. Check contract file for syntax errors:
   cat contracts/SequentialChainExecutor.sol | head -20

2. Regenerate contracts:
   python demo_sequential_composition.py --chain-type simple

3. Ensure proper contract copying:
   cp sequential_composition_output_*/*.sol contracts/
```

### ❌ `CompilerError: Stack too deep`
```bash
# Problem: Solidity function has too many local variables
# Solution: Reduce model complexity or use optimizer settings:
# In hardhat.config.js:
solidity: {
  version: "0.8.19",
  settings: {
    optimizer: {
      enabled: true,
      runs: 200
    }
  }
}
```

---

## 🚀 Deployment Errors

### ❌ `Error: could not detect network`
```bash
# Problem: Hardhat network not running
# Solutions:
1. Start Hardhat network:
   npx hardhat node --port 8545

2. Check network configuration in hardhat.config.js:
   networks: {
     localhost: {
       url: "http://127.0.0.1:8545"
     }
   }

3. Use correct network flag:
   npx hardhat run script.js --network localhost
```

### ❌ `Error: insufficient funds`
```bash
# Problem: Deployer account has no ETH
# Solutions:
1. Use Hardhat's default funded accounts:
   npx hardhat accounts

2. Check account balance:
   npx hardhat console --network localhost
   > const [deployer] = await ethers.getSigners()
   > await deployer.getBalance()

3. For custom networks, fund your account
```

### ❌ `Transaction reverted without a reason string`
```bash
# Problem: Contract constructor failed
# Solutions:
1. Check constructor parameters:
   console.log("Deploying with params:", constructorArgs)

2. Increase gas limit:
   await contract.deploy({ gasLimit: 5000000 })

3. Check for require() statements in constructor
```

### ❌ `Contract size exceeds limit`
```bash
# Problem: Contract bytecode too large (>24KB)
# Solutions:
1. Enable contract size override in hardhat.config.js:
   networks: {
     localhost: {
       allowUnlimitedContractSize: true
     }
   }

2. Use library pattern to split large contracts
3. Enable optimizer in solidity settings
```

---

## ⛓️ Sequential Execution Errors

### ❌ `Invalid step count`
```bash
# Problem: Mismatch between proof count and chain length
# Solutions:
1. Check deployment info:
   cat sequential_deployment.json

2. Verify proof data count:
   ls step*_calldata.bytes | wc -l

3. Ensure all steps deployed:
   npx hardhat console --network localhost
   > const executor = await ethers.getContractAt("SequentialChainExecutor", "0x...")
   > await executor.getChainInfo()
```

### ❌ `Step X input size mismatch`
```bash
# Problem: Data flowing between steps has wrong dimensions
# Solutions:
1. Check model input/output shapes:
   python -c "
   from chains.simple_chain_models import *
   model = ComposableFeatureExtractor()
   print('Input:', model.get_input_shape())
   print('Output:', model.get_output_shape())
   "

2. Verify chain configuration:
   npx hardhat console --network localhost
   > await executor.getStepInfo(0)

3. Check test data dimensions match:
   console.log("Test input length:", testInput.length)
```

### ❌ `Underlying verifier call failed`
```bash
# Problem: EZKL proof verification failed
# Solutions:
1. Use real proof data instead of dummy data:
   const calldata = fs.readFileSync("step00_feature_extractor_calldata.bytes")

2. Check proof format:
   npx hardhat console --network localhost
   > const verifier = await ethers.getContractAt("Halo2Verifier", underlyingAddress)
   > await verifier.verifyProof("0x" + calldata.toString("hex"))

3. Regenerate proofs if corrupted:
   python demo_sequential_composition.py --chain-type simple
```

---

## ⛽ Gas and Performance Issues

### ❌ `Transaction ran out of gas`
```bash
# Problem: Gas limit too low for sequential execution
# Solutions:
1. Increase gas limit in test:
   { gasLimit: 5000000 }

2. Estimate gas first:
   const estimate = await contract.executeSequentialChain.estimateGas(...)
   console.log("Estimated gas:", estimate.toString())

3. Use Layer 2 for lower costs:
   # Deploy on Polygon, Arbitrum, etc.
```

### ❌ `Gas price too low`
```bash
# Problem: Transaction not mined due to low gas price
# Solutions:
1. Increase gas price:
   { gasPrice: ethers.utils.parseUnits("20", "gwei") }

2. For localhost, shouldn't be an issue
3. For testnets, check current gas prices
```

### ❌ Slow proof generation
```bash
# Problem: EZKL taking too long
# Solutions:
1. Use smaller models
2. Reduce input data size
3. Use more powerful hardware
4. Run proof generation in parallel:
   # Modify pipeline to use multiprocessing
```

---

## 🔍 Debugging Commands

### Check System Status
```bash
# EZKL installation
ezkl --version

# Node/Hardhat
npx hardhat --version
node --version

# Python dependencies
python -c "import torch, ezkl; print('Dependencies OK')"
```

### Verify Blockchain State
```bash
npx hardhat console --network localhost

# Check account
> const [deployer] = await ethers.getSigners()
> await deployer.getAddress()
> await deployer.getBalance()

# Check contract deployment
> await ethers.provider.getCode("0x[contract_address]")

# Check transaction
> await ethers.provider.getTransactionReceipt("0x[tx_hash]")
```

### Debug Contract Interactions
```bash
npx hardhat console --network localhost

# Get contract instance
> const executor = await ethers.getContractAt("SequentialChainExecutor", "0x...")

# Check configuration
> await executor.getChainInfo()
> await executor.getStepInfo(0)

# Test individual calls
> const verifier = await ethers.getContractAt("EnhancedFeature_ExtractorVerifier", "0x...")
> await verifier.getContractInfo()
```

### Analyze Generated Files
```bash
# Check proof files
ls -la sequential_composition_output_*/step*_calldata.bytes

# Verify contract syntax
solc --parse-only contracts/SequentialChainExecutor.sol

# Check deployment info
cat sequential_composition_output_*/sequential_deployment.json | jq .
```

---

## 🆘 Emergency Recovery

### Complete Reset
```bash
# 1. Kill all processes
pkill -f "hardhat node"

# 2. Clean all artifacts
npx hardhat clean
rm -rf sequential_composition_output_*
rm contracts/Sequential* contracts/Enhanced* contracts/step*

# 3. Restart fresh
npx hardhat node --port 8545 &
python demo_sequential_composition.py --chain-type simple
```

### Partial Reset (Keep Proofs)
```bash
# If proofs are good but deployment failed
npx hardhat clean
npx hardhat compile
cd sequential_composition_output_*
npx hardhat run deploy_sequential_chain.js --network localhost
```

---

## 📞 Getting Help

### Debug Information to Collect
When reporting issues, include:

```bash
# System info
ezkl --version
node --version
npx hardhat --version

# Error details
cat error.log

# Contract info
ls contracts/
ls artifacts/contracts/

# Deployment info
cat sequential_composition_output_*/sequential_deployment.json

# Transaction details
npx hardhat console --network localhost
> await ethers.provider.getTransactionReceipt("0x[failed_tx]")
```

### Common Solutions Summary
1. **Restart Hardhat network** for connection issues
2. **Clean and recompile** for contract errors  
3. **Regenerate proofs** for verification failures
4. **Increase gas limits** for execution failures
5. **Check file paths** for missing artifacts

Most issues are resolved by following the complete reset procedure above. 🔧