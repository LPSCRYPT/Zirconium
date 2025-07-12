# 🚀 Zirconium: Model-Agnostic EZKL Pipeline (MAP)

**Zero-Knowledge Machine Learning Verification System**

A streamlined, model-agnostic pipeline for generating zero-knowledge proofs from any PyTorch model and verifying them on blockchain networks.

## 🎯 Key Features

- **🔄 Model-Agnostic**: Works with any PyTorch model architecture
- **⚡ Easy Switching**: Change models with a single parameter
- **🔗 Blockchain Ready**: Automatic smart contract generation and deployment
- **📊 Multi-Model Support**: RWKV, Mamba, xLSTM, MLP, and custom models
- **🛠️ Production Ready**: Complete EZKL workflow from ONNX to onchain verification
- **📈 Performance Tracking**: Gas usage and proof size comparison across models

## 🏗️ Repository Structure

```
zirconium/
├── map/                          # 🎯 Model-Agnostic Pipeline (MAP)
│   ├── core/                     # Core pipeline engine
│   │   └── ezkl_pipeline.py      # Main MAP implementation
│   ├── models/                   # Model implementations
│   │   ├── model_registry.py     # Model factory and registry
│   │   ├── weather_xlstm.py      # Weather prediction model (proven)
│   │   ├── simple_mlp.py         # Simple MLP classifier
│   │   ├── rwkv_simple.py        # RWKV language model
│   │   ├── mamba_simple.py       # Mamba state space model
│   │   └── xlstm_simple.py       # Simplified xLSTM model
│   ├── examples/                 # Usage examples and demos
│   ├── outputs/                  # Generated proofs and artifacts
│   ├── docs/                     # Documentation
│   ├── tools/                    # Utility scripts
│   ├── tests/                    # Test suites
│   └── demo.py                   # Main CLI interface
├── infrastructure/               # 🏗️ Blockchain & Deployment
│   ├── contracts/                # Smart contracts
│   ├── scripts/                  # Deployment scripts
│   ├── config/                   # Network configurations
│   └── blockchain/               # Compiled artifacts
├── archive/                      # 📚 Legacy & Experimental
│   ├── legacy-scripts/           # Old standalone scripts
│   ├── old-workflows/            # Previous implementations
│   ├── ezkl_workspace/          # Original EZKL experiments
│   └── experimental/            # Research and prototypes
└── [Root Config Files]           # Package.json, hardhat.config.js, etc.
```

## 🚀 Quick Start

### Prerequisites
```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Start local blockchain (for testing)
npx hardhat node
```

### Basic Usage

#### Run with Default Model (Weather xLSTM)
```bash
cd map
python demo.py
```

#### Switch to Different Model
```bash
python demo.py --model rwkv_simple
python demo.py --model mamba_simple
python demo.py --model simple_mlp
```

#### List Available Models
```bash
python demo.py --list
```

#### Compare All Models
```bash
python demo.py --compare
```

#### Generate Blockchain Scripts
```bash
python demo.py --model weather_xlstm --blockchain-scripts
```

### Complete Blockchain Verification
```bash
# 1. Generate proof and blockchain scripts
python demo.py --model mamba_simple --blockchain-scripts

# 2. Deploy verifier contract
npx hardhat run outputs/output_mamba_simple/deploy_mamba_simple.js --network localhost

# 3. Verify proof onchain
cd outputs/output_mamba_simple
npx hardhat run verify_mamba_simple.js --network localhost
```

## 📊 Supported Models

| Model | Architecture | Domain | Status | Gas Usage |
|-------|-------------|--------|---------|-----------|
| `weather_xlstm` | Extended LSTM | Weather Prediction | ✅ Proven | ~800K |
| `simple_mlp` | 3-layer MLP | Classification | ✅ Proven | ~537K |
| `rwkv_simple` | RWKV | Sequence Modeling | ✅ Proven | ~614K |
| `mamba_simple` | State Space Model | Sequence Modeling | ✅ Proven | ~615K |
| `xlstm_simple` | Extended LSTM | Sequence Modeling | ✅ Proven | ~616K |
| `gan_simple` | 3-layer MLP Generator | Generative Modeling | ✅ Proven | ~678K |

## 🔧 Adding Custom Models

### 1. Implement ModelInterface
```python
from map.core.ezkl_pipeline import ModelInterface

class YourModel(ModelInterface):
    def get_model(self) -> torch.nn.Module:
        return your_pytorch_model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.tensor([[...]])
    
    def get_input_data(self) -> List[List[float]]:
        return [[...]]  # EZKL JSON format
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "your_model",
            "description": "Model description",
            "architecture": "Your architecture",
            "domain": "application_domain"
        }
```

### 2. Register Your Model
```python
from map.models.model_registry import registry
registry.register("your_model", YourModel)
```

### 3. Use Your Model
```bash
python demo.py --model your_model
```

## 🎯 MAP Pipeline Architecture

The Model-Agnostic Pipeline (MAP) consists of:

1. **ModelInterface**: Abstract interface for all models
2. **EZKLPipeline**: Core pipeline engine
3. **ModelRegistry**: Factory pattern for model management
4. **Demo CLI**: User-friendly command-line interface

### Pipeline Steps
1. **ONNX Export**: Convert PyTorch model to ONNX
2. **Settings Generation**: Create EZKL configuration
3. **Circuit Compilation**: Compile arithmetic circuit
4. **Key Generation**: Create proving/verification keys
5. **Witness Generation**: Generate witness from model inference
6. **Proof Generation**: Create zero-knowledge proof
7. **Local Verification**: Verify proof locally
8. **Contract Generation**: Create Solidity verifier contract
9. **Calldata Generation**: Prepare blockchain transaction data
10. **Blockchain Deployment**: Deploy and verify onchain

## 📈 Performance Benchmarks

Real blockchain verification results:

### Gas Usage by Model Type
- **Simple MLP**: 537K gas (most efficient)
- **RWKV/Mamba/xLSTM**: 616-620K gas (similar complexity)
- **Weather xLSTM**: 799K gas (production model)

### Proof Sizes
- **Simple MLP**: ~18KB
- **Sequence Models**: ~29KB
- **Weather xLSTM**: ~32KB

## 🔗 Blockchain Integration

### Supported Networks
- **Localhost**: Hardhat development network
- **Testnets**: Sepolia, Flow Testnet, Zircuit Testnet
- **Mainnets**: Ready for production deployment

### Real Transaction Examples
All models have been successfully verified onchain:
- Weather xLSTM: `0x893fb6d...` (799K gas)
- Simple MLP: `0x91eec0c...` (537K gas)
- RWKV Simple: `0x9e9e312...` (616K gas)
- Mamba Simple: `0xd5660db...` (617K gas)
- xLSTM Simple: `0x63641da...` (618K gas)

## 📚 Documentation

- [**Quick Start Guide**](map/docs/QUICKSTART.md)
- [**Custom Models**](map/docs/CUSTOM_MODELS.md)
- [**API Reference**](map/docs/API.md)
- [**Blockchain Deployment**](map/docs/BLOCKCHAIN.md)
- [**Troubleshooting**](map/docs/TROUBLESHOOTING.md)

## 🛠️ Development

### Running Tests
```bash
cd map/tests
python -m pytest
```

### Architecture Overview
```bash
python demo.py --help
```

## 🎖️ Key Benefits

1. **🔄 Model Agnostic**: Same workflow for any PyTorch model
2. **⚡ Easy Switching**: Single parameter to test different architectures  
3. **🏗️ Production Ready**: Complete end-to-end blockchain verification
4. **📊 Performance Tracking**: Real gas measurements and proof sizes
5. **🧩 Extensible**: Easy to add new model types
6. **🔒 Zero-Knowledge**: Private model inference with public verification
7. **⛓️ Blockchain Native**: Smart contract generation and deployment

## 📜 License

[Your License Here]

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

**Perfect for rapid prototyping and testing different model architectures with zero-knowledge proofs on blockchain networks.**