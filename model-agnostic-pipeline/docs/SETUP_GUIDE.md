# Setup Guide

## Current Implementation Status

⚠️ **This system has real working components but integration gaps remain.**

**What works**: Neural networks, EZKL proof generation, smart contracts  
**What needs work**: EZKL proof parsing, environment setup, end-to-end testing

## Prerequisites

### 1. EZKL Installation
```bash
# Install Rust (required for EZKL)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install EZKL
cargo install ezkl
```

### 2. Python Dependencies
```bash
# Core dependencies
pip install -r requirements.txt

# Blockchain dependencies (for actual submission)
pip install -r requirements-blockchain.txt
```

### 3. Environment Variables
```bash
# Required for blockchain submission
export PRIVATE_KEY=your_sepolia_private_key_here

# Optional (defaults to public RPC)
export SEPOLIA_URL=https://eth-sepolia.g.alchemy.com/v2/YOUR_API_KEY
```

### 4. Testnet ETH
1. Visit https://sepoliafaucet.com/
2. Request testnet ETH for your address
3. Minimum 0.001 ETH required

## Usage

### Generate Weather Prediction with Real ZK Proof
```bash
# This will take 15-30 minutes
python src/weather_prediction.py
```

**Expected output**:
```
🌤️ Weather Prediction System Initialized
🚀 Starting Verifiable Weather Prediction
📊 Preparing Weather Data
🧠 Running xLSTM Model Inference
📦 Exporting for EZKL Proof Generation
🔐 Generating Zero-Knowledge Proof
   🔧 Running EZKL proof generation...
   ⏳ This will take 15-30 minutes...
   ⚙️ Generated EZKL settings
   🏗️ Compiled circuit
   🔑 Generated proving/verification keys
   🧮 Generated witness
   🔐 Generated ZK proof
   ✅ Proof verified successfully!
```

### Submit to Blockchain (Requires Setup)
```bash
# Set private key
export PRIVATE_KEY=your_sepolia_private_key

# Submit prediction
python src/blockchain_submission.py
```

## Known Issues

### 1. EZKL Proof Format
**Issue**: EZKL 22.0.1 proof format not properly parsed for Solidity contracts  
**Status**: Critical gap  
**Fix needed**: Update `parse_ezkl_proof()` function in `src/blockchain_submission.py`

### 2. Gas Estimation
**Issue**: Gas limit hard-coded, may fail on complex proofs  
**Status**: Minor issue  
**Fix needed**: Dynamic gas estimation based on proof complexity

### 3. Error Handling
**Issue**: Limited error handling for EZKL failures  
**Status**: Quality of life  
**Fix needed**: Better error messages and recovery

## Troubleshooting

### EZKL Installation Issues
```bash
# Check Rust installation
rustc --version

# Update Rust
rustup update

# Clean install EZKL
cargo install --force ezkl
```

### Proof Generation Failures
```bash
# Check EZKL version
ezkl --version

# Common issues:
# - Model too complex for circuit
# - Insufficient memory
# - Settings file corruption
```

### Blockchain Connection Issues
```bash
# Test connection
python -c "from web3 import Web3; print(Web3(Web3.HTTPProvider('https://rpc.sepolia.org')).is_connected())"

# Check balance
python -c "from web3 import Web3; w3=Web3(Web3.HTTPProvider('https://rpc.sepolia.org')); print(w3.from_wei(w3.eth.get_balance('YOUR_ADDRESS'), 'ether'))"
```

## Next Steps

1. **Fix EZKL parsing**: Update proof parser for current format
2. **Add error handling**: Better user experience
3. **Optimize gas**: Reduce transaction costs
4. **Add tests**: Comprehensive test suite
5. **Documentation**: Complete API docs

See [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) for detailed status.