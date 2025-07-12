# 🔧 Troubleshooting Guide

**Common issues and solutions when working with Zirconium**

## Installation Issues

### EZKL Installation Fails

**Problem**: `cargo install ezkl` fails with compilation errors

**Solutions**:
```bash
# Update Rust toolchain
rustup update stable

# Install with specific features
cargo install ezkl --features metal  # macOS only

# If still failing, install from source
git clone https://github.com/zkonduit/ezkl
cd ezkl
cargo install --path .
```

### Python Dependencies Conflict

**Problem**: `pip install -r requirements.txt` fails

**Solutions**:
```bash
# Use virtual environment (strongly recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt

# If specific package fails
pip install --upgrade pip setuptools wheel
pip install package_name --no-cache-dir
```

### Node.js Version Issues

**Problem**: `npm install` fails or contracts don't compile

**Solutions**:
```bash
# Check Node version (needs 18+)
node --version

# Use nvm to install correct version
nvm install 18
nvm use 18

# Clear npm cache if needed
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

## Proof Generation Issues

### "Failed to generate witness"

**Problem**: `ezkl gen-witness` fails

**Possible Causes & Solutions**:

1. **Invalid input format**:
   ```bash
   # Check input.json format
   cat input.json
   # Should be: {"input_data": [[val1, val2, ...]]}
   ```

2. **Model-settings mismatch**:
   ```bash
   # Regenerate settings for your model
   ezkl gen-settings --model=model.onnx --settings-path=settings.json
   ```

3. **Memory issues**:
   ```bash
   # Use smaller batch size or simpler model
   # Check available memory: free -h (Linux) or top (macOS)
   ```

### "Proof verification failed"

**Problem**: Generated proof doesn't verify

**Solutions**:
```bash
# Check if circuit compilation matches
ezkl verify --proof-path=proof.json --vk-path=vk.key --settings-path=settings.json

# Regenerate everything from scratch
ezkl gen-settings --model=model.onnx --settings-path=settings.json
ezkl compile-circuit --model=model.onnx --compiled-circuit=model.compiled --settings-path=settings.json
ezkl setup --compiled-circuit=model.compiled --vk-path=vk.key --pk-path=pk.key
```

### "Circuit too large" Error

**Problem**: Circuit exceeds memory or size limits

**Solutions**:
- **Use simpler models** (rwkv_simple instead of rwkv_complex)
- **Reduce input size** in settings.json
- **Increase system memory** or use cloud instance
- **Check model complexity**: Use smaller ONNX models

## Blockchain Integration Issues

### "Contract deployment failed"

**Problem**: `npx hardhat run scripts/deploy_minimal.js` fails

**Solutions**:
```bash
# Check if Hardhat node is running
npx hardhat node  # In separate terminal

# Verify network configuration
npx hardhat compile
npx hardhat test

# Check hardhat.config.js network settings
cat hardhat.config.js
```

### "Gas limit exceeded"

**Problem**: Proof verification transactions fail

**Solutions**:
```bash
# Use smaller proofs first
python src/simple_ezkl_test.py  # Single model test

# Check gas estimation in scripts
# Look for gasLimit settings in deployment scripts

# Verify proof size
ls -la ezkl_workspace/*/proof.json
# Should be ~57KB per proof
```

### "Invalid proof format"

**Problem**: Contract rejects EVM calldata

**Solutions**:
```bash
# Ensure calldata was generated properly
ls -la ezkl_workspace/*/calldata.txt

# Regenerate EVM calldata
cd ezkl_workspace/rwkv_simple
ezkl encode-evm-calldata --proof-path=proof.json --calldata-path=calldata.txt

# Check if wrapper contracts are deployed correctly
npx hardhat run scripts/deploy_minimal.js --network localhost
```

## Performance Issues

### Slow Proof Generation

**Problem**: Proofs take too long to generate

**Optimizations**:
```bash
# Use Metal acceleration (macOS)
# Automatically enabled if available

# Check system resources
top  # macOS/Linux
htop # Linux (if installed)

# Use smaller models during development
# rwkv_simple (~30s) vs rwkv_complex (~10min)

# Increase available memory
# Close other applications
```

### High Gas Consumption

**Problem**: Transactions cost too much gas

**Analysis**:
```bash
# Check gas usage patterns
# Single proof: ~470K gas (expected)
# 3-proof chain: ~1.2M gas (expected)

# If much higher:
# - Check proof sizes (should be ~57KB)
# - Verify wrapper contracts are being used
# - Ensure real EZKL verification (not mock)
```

## Network Connection Issues

### "Cannot connect to localhost"

**Problem**: Scripts can't connect to local blockchain

**Solutions**:
```bash
# Verify Hardhat node is running
curl -X POST -H "Content-Type: application/json" \
  --data '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' \
  http://localhost:8545

# Check port conflicts
lsof -i :8545  # See what's using port 8545

# Restart Hardhat node
pkill -f "hardhat node"
npx hardhat node
```

### Testnet Connection Issues

**Problem**: Cannot deploy to testnets

**Solutions**:
```bash
# Check environment variables
echo $INFURA_API_KEY
echo $PRIVATE_KEY

# Verify network configuration in hardhat.config.js
# Test connection
npx hardhat run scripts/deploy_minimal.js --network sepolia
```

## Model-Specific Issues

### ONNX Model Loading Fails

**Problem**: EZKL cannot load your ONNX model

**Solutions**:
```bash
# Check ONNX model validity
python -c "import onnx; model = onnx.load('model.onnx'); onnx.checker.check_model(model)"

# Ensure model uses supported operations
# See EZKL documentation for supported ops

# Try with provided working models first
cd ezkl_workspace/rwkv_simple
ezkl verify --proof-path=proof.json --vk-path=vk.key --settings-path=settings.json
```

### Input Format Mismatch

**Problem**: Model expects different input format

**Solutions**:
```bash
# Check expected input shape in ONNX model
python -c "import onnx; model = onnx.load('model.onnx'); print(model.graph.input)"

# Adjust input.json format accordingly
# Example: {"input_data": [[1.0, 2.0, 3.0]]}  # Single batch
```

## Getting Help

### Debug Information Collection

When reporting issues, include:

```bash
# System info
uname -a
node --version
python --version
ezkl --version

# Project state
ls -la ezkl_workspace/
cat hardhat.config.js | grep networks -A 10
git status
```

### Log Analysis

```bash
# Enable verbose logging
export EZKL_LOG=debug
ezkl gen-witness --model=model.onnx --input=input.json --output=witness.json

# Check Hardhat logs
npx hardhat node --verbose

# Python script debugging
python -u src/blockchain_proof_chain_submission.py
```

### Common Error Patterns

| Error Message | Likely Cause | Quick Fix |
|---------------|--------------|-----------|
| "Circuit compilation failed" | ONNX model incompatible | Use provided models first |
| "Witness generation failed" | Input format wrong | Check input.json structure |
| "Gas estimation failed" | Contract not deployed | Run deploy script |
| "Proof verification failed" | Keys/circuit mismatch | Regenerate keys |
| "Cannot connect to node" | Hardhat not running | Start `npx hardhat node` |

### Still Stuck?

1. **Check existing issues** in the repository
2. **Review documentation** in `/docs` folder
3. **Try with provided examples** first
4. **Compare with working state** in `examples/`