# 🎯 Custom Model Integration Guide

**Complete walkthrough for adding your own ONNX models to Zirconium**

> ⚠️ **Prerequisites**: Complete the [QUICKSTART.md](../QUICKSTART.md) guide first to ensure your environment is properly configured.

## 🎯 Overview

This guide covers the complete process of integrating your custom ONNX model into the Zirconium zkML proof chaining system, from model preparation to blockchain deployment.

### What You'll Build
- ✅ **EZKL-compatible workspace** for your model
- ✅ **Smart contract verifier** for onchain proof verification  
- ✅ **Integration scripts** for proof generation and submission
- ✅ **End-to-end validation** of your custom model pipeline

## 📋 Requirements

### Model Requirements
- **Format**: ONNX model file (`.onnx`)
- **Input Shape**: Fixed input dimensions (preferably 10-dimensional for chaining)
- **Operations**: [EZKL-supported operations](https://docs.ezkl.xyz/supported_operations) only
- **Size**: Reasonable complexity (start with simple models)

### System Requirements
- **Memory**: 8GB+ RAM (16GB recommended for complex models)
- **EZKL**: Version 7.0+ installed (`cargo install ezkl`)
- **Node.js**: v18+ for smart contract deployment
- **Python**: 3.9+ with dependencies from `requirements.txt`

## 🚀 Step-by-Step Integration

### Step 1: Prepare Your Model Workspace

```bash
# Create workspace for your model
mkdir -p ezkl_workspace/my_model
cd ezkl_workspace/my_model

# Copy your ONNX model
cp /path/to/your/model.onnx model.onnx

# Verify ONNX model structure
python -c "import onnx; model = onnx.load('model.onnx'); print([i.name for i in model.graph.input])"
```

### Step 2: Create Input Data Format

```bash
# Create input.json with your model's expected format
cat > input.json << 'EOF'
{
  "input_data": [
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
  ]
}
EOF

# Adjust the input dimensions to match your model's requirements
```

### Step 3: Generate EZKL Settings

```bash
# Generate optimized settings for your model
ezkl gen-settings \
  --model=model.onnx \
  --settings-path=settings.json \
  --input=input.json

# Optional: Adjust settings for performance/memory trade-offs
# Edit settings.json to modify:
# - "run_args": {"tolerance": {"val": 0, "scale": 1}} for precision
# - "num_inner_cols": 2 (reduce for memory, increase for speed)
```

### Step 4: Compile and Setup Circuit

```bash
# Compile neural network to arithmetic circuit
ezkl compile-circuit \
  --model=model.onnx \
  --compiled-circuit=model.compiled \
  --settings-path=settings.json

# Generate proving and verification keys
ezkl setup \
  --compiled-circuit=model.compiled \
  --vk-path=vk.key \
  --pk-path=pk.key

# Test witness generation
ezkl gen-witness \
  --compiled-circuit=model.compiled \
  --input=input.json \
  --output=witness.json
```

### Step 5: Generate and Verify Proof

```bash
# Generate cryptographic proof
ezkl prove \
  --compiled-circuit=model.compiled \
  --pk-path=pk.key \
  --proof-path=proof.json \
  --witness-path=witness.json

# Verify proof offchain
ezkl verify \
  --proof-path=proof.json \
  --vk-path=vk.key \
  --settings-path=settings.json

# Generate EVM calldata for blockchain submission
ezkl encode-evm-calldata \
  --proof-path=proof.json \
  --calldata-path=calldata.txt
```

### Step 6: Create Smart Contract Verifier

```bash
# Generate base EZKL verifier contract
ezkl create-evm-verifier \
  --vk-path=vk.key \
  --sol-code-path=MyModelVerifier.sol \
  --abi-path=MyModelVerifier.abi

# Move to contracts directory
mv MyModelVerifier.sol ../../contracts/verifiers/
```

### Step 7: Create Wrapper Contract

Create `contracts/verifiers/MyModelVerifierWrapper.sol`:

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../base/BaseEZKLVerifier.sol";
import "./MyModelVerifier.sol";

contract MyModelVerifierWrapper is BaseEZKLVerifier {
    MyModelVerifier private immutable ezkl_verifier;
    
    constructor(address _ezklVerifierAddress) {
        ezkl_verifier = MyModelVerifier(_ezklVerifierAddress);
    }
    
    function verify(bytes memory proof, uint256[] memory publicInputs) 
        public override returns (bool) {
        
        // Validate inputs
        require(proof.length > 0, "Empty proof");
        require(publicInputs.length >= 1, "Insufficient public inputs");
        
        try ezkl_verifier.verifyProof(proof, publicInputs) {
            // Successful verification
            totalVerifications++;
            bytes32 proofHash = keccak256(proof);
            
            emit ProofVerified(proofHash, address(this), true);
            return true;
            
        } catch Error(string memory reason) {
            // Handle verification failure
            bytes32 proofHash = keccak256(proof);
            emit ProofVerified(proofHash, address(this), false);
            revert(string.concat("Verification failed: ", reason));
            
        } catch {
            // Handle unexpected errors
            bytes32 proofHash = keccak256(proof);
            emit ProofVerified(proofHash, address(this), false);
            revert("Verification failed: Unknown error");
        }
    }
    
    function getArchitecture() public pure override returns (string memory) {
        return "MyModel";  // Replace with your model name
    }
}
```

### Step 8: Update Deployment Script

Add your model to `scripts/deploy_minimal.js`:

```javascript
// Add to the deployment function
const MyModelVerifier = await ethers.getContractFactory("MyModelVerifier");
const myModelVerifier = await MyModelVerifier.deploy();
await myModelVerifier.waitForDeployment();

const MyModelVerifierWrapper = await ethers.getContractFactory("MyModelVerifierWrapper");
const myModelVerifierWrapper = await MyModelVerifierWrapper.deploy(
    await myModelVerifier.getAddress()
);
await myModelVerifierWrapper.waitForDeployment();

// Add to contracts object
contracts.MyModelVerifier = await myModelVerifier.getAddress();
contracts.MyModelVerifierWrapper = await myModelVerifierWrapper.getAddress();
```

### Step 9: Deploy and Test

```bash
# Compile contracts
npx hardhat compile

# Deploy to localhost (make sure hardhat node is running)
npx hardhat run scripts/deploy_minimal.js --network localhost

# Test your model integration
cd ../../
python -c "
from src.blockchain_proof_chain_submission import BlockchainProofChainSubmitter
submitter = BlockchainProofChainSubmitter('localhost')
result = submitter.submit_individual_proof('MyModel', 'ezkl_workspace/my_model/proof.json')
print('Success!' if result['success'] else f'Failed: {result.get(\"error\")}')
"
```

## 🔧 Integration into Proof Chains

### Option 1: Replace Existing Model

Modify `src/proof_chaining.py` to use your model instead of one of the existing models:

```python
# Replace in the models list
self.models = ["rwkv_simple", "my_model", "xlstm_simple"]  # Replace mamba_simple
```

### Option 2: Extend Chain

Add your model as a new step in the chain:

```python
# Extend the pipeline
self.models = ["rwkv_simple", "mamba_simple", "xlstm_simple", "my_model"]
```

### Option 3: Custom Chain

Create a completely custom proof chain:

```python
# In proof_chaining.py
def run_custom_chain(self):
    """Custom proof chain with your models"""
    models = ["my_model_1", "my_model_2", "my_model_3"]
    # Implementation follows existing pattern...
```

## 🎯 Model-Specific Considerations

### Input/Output Formatting

Ensure your model follows the chaining format:
- **Input**: 10-dimensional array for compatibility
- **Output**: At least 10 values for chaining to next model

```python
# In your model preprocessing
def prepare_chained_input(previous_output):
    """Convert previous model output to your model's input format"""
    # Take first 10 values or reshape as needed
    return previous_output[:10]

def extract_chainable_output(model_output):
    """Extract first 10 values for next model in chain"""
    return model_output[:10]
```

### Performance Optimization

```bash
# For faster proof generation
# 1. Reduce circuit size in settings.json
{
  "run_args": {
    "tolerance": {"val": 2, "scale": 1}  # Increase tolerance
  },
  "num_inner_cols": 1  # Reduce for memory efficiency
}

# 2. Use smaller input dimensions when possible
# 3. Simplify ONNX model (remove unnecessary operations)
```

### Memory Management

```python
# Monitor memory usage during development
import psutil
import os

def check_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")

# Call before/after EZKL operations
check_memory()
```

## 🐛 Troubleshooting Common Issues

### Circuit Compilation Fails

```bash
# Check ONNX model compatibility
python -c "
import onnx
model = onnx.load('model.onnx')
print('Supported ops:', [node.op_type for node in model.graph.node])
"

# Compare with EZKL supported operations list
# Simplify model if using unsupported operations
```

### Proof Generation Too Slow

```bash
# Use smaller models during development
# Optimize settings.json
# Check system resources (htop/Activity Monitor)
# Consider using cloud instances with more RAM
```

### Contract Size Too Large

```bash
# Check contract size
npx hardhat compile
du -h artifacts/contracts/verifiers/MyModelVerifier.sol/MyModelVerifier.json

# If >24KB, deploy only to localhost/testnets
# Consider L2 solutions for mainnet deployment
```

### Gas Estimation Failures

```python
# Reduce proof size or simplify model
# Check if contract was deployed correctly
# Verify calldata format matches contract expectations
```

## 📊 Performance Benchmarks

### Expected Metrics by Model Complexity

| Model Type | Circuit Size | Proof Time | Proof Size | Gas Cost |
|------------|-------------|------------|------------|----------|
| Simple (10→10) | ~350K gates | 30-60s | ~57KB | ~470K |
| Medium (50→50) | ~1M gates | 2-5min | ~65KB | ~500K |
| Complex (100→100) | ~3M gates | 10-30min | ~80KB | ~600K |

### Optimization Guidelines

- **Start simple**: Begin with small input dimensions
- **Profile incrementally**: Add complexity gradually  
- **Monitor resources**: Check memory/time usage
- **Test locally first**: Validate before testnet deployment

## 🎯 Next Steps

1. **Successfully integrate** one custom model using this guide
2. **Experiment with chaining** your model with existing ones
3. **Optimize performance** for your specific use case
4. **Consider production deployment** to testnets
5. **Explore advanced features** like model versioning and marketplaces

## 📚 Additional Resources

- **EZKL Documentation**: https://docs.ezkl.xyz/
- **ONNX Tutorials**: https://onnx.ai/onnx/tutorials/
- **Hardhat Guides**: https://hardhat.org/docs
- **Troubleshooting**: [TROUBLESHOOTING.md](./TROUBLESHOOTING.md)
- **Local Development**: [LOCAL_DEVELOPMENT.md](./LOCAL_DEVELOPMENT.md)

---

🎉 **Successfully integrated your custom model?** Consider contributing your model as an example to help other developers!