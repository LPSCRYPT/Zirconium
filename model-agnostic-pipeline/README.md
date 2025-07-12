# 🎯 Model-Agnostic Pipeline (MAP)

**The Core Engine for Zero-Knowledge Machine Learning**

MAP provides a unified interface for converting any PyTorch model into a blockchain-verifiable zero-knowledge proof system.

## 🚀 Quick Start

### Run Demo with Default Model
```bash
python demo.py
```

### Switch Between Models
```bash
python demo.py --model rwkv_simple    # RWKV language model
python demo.py --model mamba_simple   # Mamba state space model  
python demo.py --model simple_mlp     # Simple MLP classifier
python demo.py --model weather_xlstm  # Weather prediction (proven)
```

### Compare All Models
```bash
python demo.py --compare
```

### Generate Blockchain Scripts
```bash
python demo.py --model mamba_simple --blockchain-scripts
```

## 🏗️ Architecture

### Core Components

1. **`core/ezkl_pipeline.py`** - Main pipeline engine
   - `ModelInterface` - Abstract interface for all models
   - `EZKLPipeline` - Complete EZKL workflow implementation

2. **`models/model_registry.py`** - Model management
   - Factory pattern for model instantiation
   - Automatic model registration
   - Easy model switching

3. **`demo.py`** - Command-line interface
   - User-friendly CLI for all operations
   - Automatic script generation
   - Performance comparison tools

### Workflow Steps

The MAP executes these steps for any model:

1. **Model Export** → ONNX conversion
2. **Input Generation** → JSON input data
3. **Settings Creation** → EZKL configuration
4. **Circuit Compilation** → Arithmetic circuit
5. **Key Generation** → Proving/verification keys
6. **Witness Generation** → Model inference witness
7. **Proof Generation** → Zero-knowledge proof
8. **Local Verification** → Proof validation
9. **Contract Generation** → Solidity verifier
10. **Calldata Generation** → Blockchain transaction data

## 📊 Supported Models

| Model | Architecture | Domain | Status | Gas Usage |
|-------|-------------|---------|---------|-----------|
| `weather_xlstm` | Extended LSTM | Weather Prediction | ✅ Proven | ~800K |
| `simple_mlp` | 3-layer MLP | Classification | ✅ Proven | ~537K |
| `rwkv_simple` | RWKV | Sequence Modeling | ✅ Proven | ~614K |
| `mamba_simple` | State Space Model | Sequence Modeling | ✅ Proven | ~615K |
| `xlstm_simple` | Extended LSTM | Sequence Modeling | ✅ Proven | ~616K |
| `gan_simple` | 3-layer MLP Generator | Generative Modeling | ✅ Proven | ~678K |
| `gan_compact` | 4-layer MLP Generator | Generative Modeling | ✅ Proven | ~1.8M |

## 🔧 Adding New Models

### 1. Create Model Class

```python
# models/your_model.py
from core.ezkl_pipeline import ModelInterface
import torch
import torch.nn as nn

class YourModelArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # Your model definition
        
    def forward(self, x):
        # Your forward pass
        return x

class YourModel(ModelInterface):
    def get_model(self) -> torch.nn.Module:
        model = YourModelArchitecture()
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, 10)  # Your input shape
    
    def get_input_data(self) -> List[List[float]]:
        return [[1.0, 2.0, 3.0, ...]]  # EZKL format
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "your_model",
            "description": "Your model description",
            "architecture": "Your Architecture Type",
            "domain": "your_domain",
            "status": "🧪 EXPERIMENTAL"
        }
```

### 2. Register Model

```python
# In models/model_registry.py, add to imports:
from your_model import YourModel

# In _register_default_models():
self.register("your_model", YourModel)
```

### 3. Use Model

```bash
python demo.py --model your_model
```

## 📁 Output Structure

Each model run creates an output directory:

```
outputs/output_<model_name>/
├── <model_name>.onnx           # Exported ONNX model
├── input.json                  # Input data for EZKL
├── settings.json               # EZKL settings
├── model.compiled              # Compiled circuit
├── pk.key                      # Proving key
├── vk.key                      # Verification key
├── witness.json                # Generated witness
├── proof.json                  # Zero-knowledge proof
├── <model_name>Verifier.sol    # Solidity verifier contract
├── <model_name>Verifier.json   # Contract ABI
├── calldata.bytes              # Blockchain calldata
├── deploy_<model_name>.js      # Deployment script
└── verify_<model_name>.js      # Verification script
```

## 🔗 Blockchain Integration

⚠️ **IMPORTANT**: First-time setup required. See [SETUP.md](SETUP.md) for complete instructions.

### Quick Setup (One-time)
```bash
npm init -y
npm install --save-dev hardhat @nomicfoundation/hardhat-ethers ethers
# Create hardhat.config.js (see SETUP.md)
mkdir contracts
```

### Deploy and Verify
```bash
# 1. Generate model and copy contract
python demo.py --model simple_mlp --blockchain-scripts
cp output_simple_mlp/simple_mlpVerifier.sol contracts/

# 2. Compile contracts
npx hardhat compile

# 3. Start blockchain (separate terminal)
npx hardhat node --port 8545

# 4. Deploy and verify
npx hardhat run output_simple_mlp/deploy_simple_mlp.js --network localhost
npx hardhat run output_simple_mlp/verify_simple_mlp.js --network localhost
```

## 📈 Performance Benchmarks

### Gas Usage (Localhost) - All PROVEN ✅
- **Simple MLP**: 537K gas ✅ VERIFIED
- **RWKV Simple**: 614K gas ✅ VERIFIED  
- **Mamba Simple**: 615K gas ✅ VERIFIED
- **xLSTM Simple**: 616K gas ✅ VERIFIED
- **Weather xLSTM**: 800K gas ✅ VERIFIED
- **GAN Simple**: 678K gas ✅ VERIFIED (8×8 grayscale)
- **GAN Compact**: 1.8M gas ✅ VERIFIED (16×16 RGB)

### Proof Sizes
- **Simple MLP**: 18KB
- **Sequence Models**: 29KB
- **Weather xLSTM**: 32KB
- **GAN Models**: ~15-25KB

### Generation Times
- **Model Export**: ~1-2 seconds
- **Circuit Compilation**: ~10-30 seconds
- **Proof Generation**: ~30-60 seconds
- **Total Pipeline**: ~1-2 minutes

## 🛠️ Development

### Adding Features
1. Extend `ModelInterface` for new capabilities
2. Update `EZKLPipeline` for new pipeline steps
3. Modify `demo.py` for new CLI options

### Testing New Models
```bash
# Test model export
python -c "from models.your_model import YourModel; m = YourModel(); print(m.get_config())"

# Run full pipeline
python demo.py --model your_model

# Test blockchain verification  
python demo.py --model your_model --blockchain-scripts
```

## 📚 Documentation

- [Complete Setup Guide](SETUP.md) ⚠️ **Required for blockchain verification**
- [Troubleshooting Guide](TROUBLESHOOTING.md) 🔧 **Read this if you encounter issues**
- [API Reference](docs/API.md)
- [Custom Models Guide](docs/CUSTOM_MODELS.md)
- [Blockchain Integration](docs/BLOCKCHAIN.md)

## 🎯 Design Principles

1. **Model Agnostic**: Works with any PyTorch model
2. **Consistent Interface**: Same API for all models
3. **Automatic Generation**: Minimal manual configuration
4. **Blockchain Native**: Built for onchain verification
5. **Developer Friendly**: Easy to extend and modify

---

**The MAP makes zero-knowledge machine learning accessible to any PyTorch model with a simple, unified interface.**