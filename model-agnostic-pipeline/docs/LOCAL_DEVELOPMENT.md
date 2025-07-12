# 🛠️ Local Development Setup

**Comprehensive guide for developers working on Zirconium**

## System Requirements

### Required Software
- **Node.js** v18.0+ (LTS recommended)
- **Python** 3.9-3.11 (3.12+ not yet supported by some dependencies)
- **Rust** (latest stable for EZKL)
- **Git** 
- **Make** (for using Makefile commands)

### Hardware Requirements
- **Memory**: 8GB RAM minimum (16GB recommended for complex models)
- **Storage**: 2GB free space for dependencies and artifacts
- **OS**: macOS, Linux, or WSL2 on Windows

## Installation Steps

### 1. Install EZKL (Required)

```bash
# Install Rust if not already installed
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install EZKL
cargo install ezkl

# Verify installation
ezkl --version
# Should output: ezkl 0.x.x
```

### 2. Clone and Setup Project

```bash
git clone <repository-url>
cd zirconium

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Verify setup
make check-deps
```

### 3. Project Structure Overview

```
zirconium/
├── contracts/          # Solidity smart contracts
├── src/               # Python scripts for proof generation
├── ezkl_workspace/    # EZKL model compilation workspace
├── scripts/           # Deployment and interaction scripts
├── test/              # Contract tests
├── docs/              # Documentation
└── examples/          # Usage examples
```

## Development Workflow

### Local Blockchain Setup

```bash
# Terminal 1: Start local Hardhat node
npx hardhat node

# Terminal 2: Deploy contracts (keep this terminal for commands)
npx hardhat run scripts/deploy_minimal.js --network localhost
```

### Working with Models

```bash
# Compile a new model
cd ezkl_workspace/rwkv_simple
ezkl compile-circuit --model=model.onnx --compiled-circuit=model.compiled --settings-path=settings.json

# Generate proof
ezkl prove --compiled-circuit=model.compiled --proof-path=proof.json --witness-path=witness.json

# Verify proof
ezkl verify --proof-path=proof.json --vk-path=vk.key --settings-path=settings.json
```

### Testing Your Changes

```bash
# Run contract tests
npx hardhat test

# Run integration tests
python test_integration.py

# Test specific proof chain
python src/blockchain_proof_chain_submission.py

# Run offchain demo
python examples/offchain_proof_chain_demo.py
```

## Common Development Tasks

### Adding a New Model

1. **Place ONNX model** in `ezkl_workspace/your_model/`
2. **Create settings.json** using `ezkl gen-settings`
3. **Compile circuit** with `ezkl compile-circuit`
4. **Setup keys** with `ezkl setup` and `ezkl gen-witness`
5. **Generate verifier contract** with `ezkl create-evm-verifier`
6. **Create wrapper contract** following existing patterns

### Modifying Proof Chain Logic

- **Edit** `src/proof_chaining.py` for offchain logic
- **Edit** `src/blockchain_proof_chain_submission.py` for onchain submission
- **Test changes** with local blockchain

### Debugging Failed Proofs

```bash
# Check witness generation
ezkl gen-witness --compiled-circuit=model.compiled --input=input.json --output=witness.json

# Verify circuit compilation
ezkl compile-circuit --model=model.onnx --compiled-circuit=test.compiled --settings-path=settings.json

# Check proof generation step by step
ezkl prove --compiled-circuit=model.compiled --proof-path=test_proof.json --witness-path=witness.json --proving-key=pk.key
```

## Environment Configuration

### Required Environment Variables

```bash
# Add to your .bashrc or .zshrc
export NODE_ENV=development
export HARDHAT_NETWORK=localhost

# Optional: For testnet deployment
export INFURA_API_KEY=your_key_here
export PRIVATE_KEY=your_private_key_here
```

### Python Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## IDE Setup Recommendations

### VS Code Extensions
- **Solidity** (Juan Blanco)
- **Python** (Microsoft)  
- **Hardhat Solidity** (Nomic Foundation)
- **Rust-analyzer** (for EZKL development)

### VS Code Settings
```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "solidity.compileUsingRemoteVersion": "0.8.24"
}
```

## Performance Tips

### Faster Proof Generation
- **Use smaller models** during development (rwkv_simple vs rwkv_complex)
- **Enable Metal acceleration** on macOS (automatic if available)
- **Increase RAM allocation** for large circuits

### Gas Optimization
- **Test on localhost first** before testnets
- **Use gas estimation** in scripts
- **Batch multiple operations** when possible

## Next Steps

- Read `docs/APPLICATIONS.md` for use cases
- Check `examples/` for advanced usage patterns
- See `setup/` for testnet deployment guides
- Review `docs/FORMALISMS.md` for mathematical background