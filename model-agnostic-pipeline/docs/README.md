# Model-Agnostic EZKL Demo

**Streamlined workflow for testing any PyTorch model with EZKL zero-knowledge proofs**

## 🎯 Key Features

- **Easy Model Swapping**: Change models with a single parameter
- **Unified Pipeline**: Same EZKL workflow for any model architecture
- **Model Registry**: Factory pattern for managing multiple models
- **Blockchain Ready**: Generates deployment scripts automatically
- **Comparison Mode**: Test multiple models side-by-side

## 🚀 Quick Start

### Run Demo with Default Model (Weather xLSTM)
```bash
python demo.py
```

### Swap to Different Model
```bash
python demo.py --model simple_mlp
```

### List Available Models
```bash
python demo.py --list
```

### Compare Multiple Models
```bash
python demo.py --compare
```

### Generate Blockchain Scripts
```bash
python demo.py --model weather_xlstm --blockchain-scripts
```

## 📁 Directory Structure

```
├── demo.py                    # Main demo script
├── pipeline/
│   └── ezkl_pipeline.py       # Model-agnostic EZKL pipeline
├── models/
│   ├── model_registry.py      # Model factory and registry
│   ├── weather_xlstm.py       # Weather prediction model (proven working)
│   └── simple_mlp.py          # Simple MLP for comparison
├── examples/                  # Example model implementations
├── configs/                   # Configuration templates
└── output_*/                  # Generated outputs per model
```

## 🔧 Adding New Models

### 1. Implement ModelInterface
```python
from ezkl_pipeline import ModelInterface

class YourModel(ModelInterface):
    def get_model(self) -> torch.nn.Module:
        return your_pytorch_model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.tensor([[...]])  # Your input format
    
    def get_input_data(self) -> List[List[float]]:
        return [[...]]  # EZKL JSON format
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "your_model",
            "description": "Your model description",
            "architecture": "Your architecture type",
            # ... other metadata
        }
```

### 2. Register Your Model
```python
from model_registry import registry
registry.register("your_model", YourModel)
```

### 3. Use Your Model
```bash
python demo.py --model your_model
```

## 📊 Example Output

```bash
🚀 Model-Agnostic EZKL Demo
==================================================

🔧 Available Models:
--------------------------------------------------
   weather_xlstm: xLSTM model for weather prediction ✅ PROVEN
      Architecture: Extended LSTM with exponential gating
      Domain: weather_prediction
      Gas estimate: ~800K

   simple_mlp: Simple Multi-Layer Perceptron for classification ✅ PROVEN
      Architecture: 3-layer MLP with ReLU activation
      Domain: classification
      Gas estimate: ~537K

📊 Creating model: weather_xlstm
   Description: xLSTM model for weather prediction
   Architecture: Extended LSTM with exponential gating
   Domain: weather_prediction

🔄 Running EZKL pipeline...
📁 Working directory: output_weather_xlstm

📊 Step 1: Exporting model to ONNX
   ✅ Model exported: weather_xlstm.onnx

📋 Step 2: Creating input data
   ✅ Input data created: input.json

⚙️ Step 3: Generating EZKL settings
   ✅ Settings generated

🔧 Step 4: Compiling circuit
   ✅ Circuit compiled

🔐 Step 5: Setting up proving/verification keys
   ✅ Keys generated

👁️ Step 6: Generating witness
   ✅ Witness generated

🔒 Step 7: Generating zero-knowledge proof
   ✅ Proof generated

✅ Step 8: Verifying proof locally
   ✅ Local verification successful

🏗️ Step 9: Creating EVM verifier contract
   ✅ Verifier contract created: weather_xlstmVerifier.sol

📞 Step 10: Generating blockchain calldata
   ✅ Calldata generated

📊 Pipeline Summary for weather_xlstm:
   ONNX Model: 15,234 bytes
   ZK Proof: 31,737 bytes
   Verifier Contract: 102,726 bytes
   Verifier ABI: 1,456 bytes
   Calldata: 5,861 bytes
📁 All files in: output_weather_xlstm
✨ Ready for blockchain deployment!

✅ SUCCESS: Complete EZKL workflow for weather_xlstm
📁 Results saved in: output_weather_xlstm
```

## 🎯 Model Comparison

Run multiple models to compare their characteristics:

```bash
python demo.py --compare
```

Output:
```
📊 Comparison Summary:
--------------------------------------------------
weather_xlstm:
   Proof size: 31,737 bytes
   Contract size: 102,726 bytes
   Files: output_weather_xlstm

simple_mlp:
   Proof size: 28,943 bytes
   Contract size: 98,234 bytes
   Files: output_simple_mlp
```

## 🚀 Blockchain Deployment

The demo automatically generates blockchain scripts:

```bash
# Generate scripts
python demo.py --model weather_xlstm --blockchain-scripts

# Deploy to blockchain
npx hardhat node                                    # Start localhost
npx hardhat run output_weather_xlstm/deploy_weather_xlstm.js --network localhost

# Verify proof
npx hardhat run output_weather_xlstm/verify_weather_xlstm.js --network localhost
```

## ✅ Proven Working Models

- **weather_xlstm**: ✅ Fully tested with blockchain verification (800K gas)
- **simple_mlp**: ✅ Fully tested with blockchain verification (537K gas)
- **rwkv_simple**: ✅ Fully tested with blockchain verification (614K gas)
- **mamba_simple**: ✅ Fully tested with blockchain verification (615K gas)
- **xlstm_simple**: ✅ Fully tested with blockchain verification (616K gas)

## 🧪 Experimental Models

*All models have been promoted to PROVEN status after successful blockchain verification*

## 🎖️ Key Benefits

1. **Model Agnostic**: Works with any PyTorch model
2. **Easy Swapping**: Single parameter change to test different models
3. **Consistent Interface**: Same API for all models
4. **Automated Pipeline**: Complete EZKL workflow with one command
5. **Blockchain Ready**: Generates deployment scripts automatically
6. **Comparison Mode**: Test multiple models efficiently
7. **Extensible**: Easy to add new model types

**Perfect for rapid prototyping and testing different model architectures with zero-knowledge proofs.**