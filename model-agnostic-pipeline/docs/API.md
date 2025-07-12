# 📚 MAP API Reference

## Core Classes

### ModelInterface

Abstract base class that all models must implement.

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch

class ModelInterface(ABC):
    """Abstract interface for all models in the MAP system"""
    
    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return the PyTorch model instance"""
        pass
    
    @abstractmethod
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input tensor for ONNX export"""
        pass
    
    @abstractmethod
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL JSON format"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration metadata"""
        pass
```

#### Required Methods

##### `get_model() -> torch.nn.Module`
- **Purpose**: Return the PyTorch model instance
- **Requirements**: Model must be in eval mode
- **Example**:
```python
def get_model(self) -> torch.nn.Module:
    model = MyModel()
    model.eval()
    return model
```

##### `get_sample_input() -> torch.Tensor`
- **Purpose**: Provide sample input for ONNX export
- **Requirements**: Must match model's expected input shape
- **Example**:
```python
def get_sample_input(self) -> torch.Tensor:
    return torch.randn(1, 10)  # Batch size 1, 10 features
```

##### `get_input_data() -> List[List[float]]`
- **Purpose**: Provide actual input data for proof generation
- **Requirements**: Must be in EZKL JSON format (nested lists)
- **Example**:
```python
def get_input_data(self) -> List[List[float]]:
    return [[1.0, 2.0, 3.0, 4.0, 5.0]]  # Single sample
```

##### `get_config() -> Dict[str, Any]`
- **Purpose**: Return model metadata and configuration
- **Required Keys**: `name`, `description`, `architecture`, `domain`
- **Example**:
```python
def get_config(self) -> Dict[str, Any]:
    return {
        "name": "my_model",
        "description": "A sample model for classification",
        "architecture": "3-layer MLP",
        "domain": "classification",
        "status": "🧪 EXPERIMENTAL",
        "input_size": 10,
        "output_size": 3,
        "gas_estimate": "~500K"
    }
```

### EZKLPipeline

Core pipeline engine that orchestrates the entire EZKL workflow.

```python
class EZKLPipeline:
    def __init__(self, model_interface: ModelInterface, 
                 output_dir: str = None,
                 input_visibility: str = "public", 
                 output_visibility: str = "public"):
        """Initialize EZKL pipeline"""
```

#### Constructor Parameters

- **`model_interface`**: Instance of ModelInterface implementation
- **`output_dir`**: Directory for generated files (default: `output_{model_name}`)
- **`input_visibility`**: EZKL input visibility ("public" or "private")
- **`output_visibility`**: EZKL output visibility ("public" or "private")

#### Methods

##### `run_complete_pipeline() -> bool`
- **Purpose**: Execute the full 10-step EZKL workflow
- **Returns**: True if successful, False otherwise
- **Steps**:
  1. Export model to ONNX
  2. Create input data JSON
  3. Generate EZKL settings
  4. Compile circuit
  5. Setup proving/verification keys
  6. Generate witness
  7. Generate proof
  8. Verify proof locally
  9. Create EVM verifier contract
  10. Generate blockchain calldata

##### `export_model() -> bool`
Export PyTorch model to ONNX format.

##### `create_input_data() -> bool`
Generate input.json file for EZKL.

##### `generate_settings() -> bool`
Create EZKL settings.json configuration.

##### `compile_circuit() -> bool`
Compile the arithmetic circuit.

##### `setup_keys() -> bool`
Generate proving and verification keys.

##### `generate_witness() -> bool`
Create witness from model inference.

##### `generate_proof() -> bool`
Generate the zero-knowledge proof.

##### `verify_proof() -> bool`
Verify the proof locally.

##### `create_verifier_contract() -> bool`
Generate Solidity verifier contract.

##### `generate_calldata() -> bool`
Create blockchain transaction calldata.

### ModelRegistry

Factory for managing and creating model instances.

```python
class ModelRegistry:
    def register(self, name: str, model_class: Type[ModelInterface]):
        """Register a new model class"""
    
    def create_model(self, name: str, **kwargs) -> ModelInterface:
        """Create model instance by name"""
    
    def list_models(self) -> List[str]:
        """List all registered model names"""
    
    def get_model_info(self, name: str) -> Dict:
        """Get model configuration info"""
```

#### Global Registry

```python
from models.model_registry import registry

# Register new model
registry.register("my_model", MyModelClass)

# Create model instance
model = registry.create_model("my_model")

# List available models
models = registry.list_models()
```

## CLI Interface

### Command Line Usage

```bash
python demo.py [options]
```

### Options

#### Model Selection
- `--model <name>`: Specify model to use (default: weather_xlstm)
- `--list`: List all available models

#### Pipeline Operations
- `--blockchain-scripts`: Generate deployment and verification scripts
- `--compare`: Run pipeline for all models and compare results

#### Examples

```bash
# Run default model
python demo.py

# Run specific model
python demo.py --model rwkv_simple

# Generate blockchain scripts
python demo.py --model mamba_simple --blockchain-scripts

# Compare all models
python demo.py --compare

# List available models
python demo.py --list
```

## Configuration

### EZKL Settings

The pipeline automatically generates EZKL settings, but you can customize:

```python
# In your model's get_config() method
def get_config(self) -> Dict[str, Any]:
    return {
        # ... other config ...
        "ezkl_settings": {
            "input_visibility": "public",
            "output_visibility": "public", 
            "param_visibility": "private"
        }
    }
```

### Output Directory Structure

```
outputs/output_<model_name>/
├── <model_name>.onnx           # Exported model
├── input.json                  # Input data
├── settings.json               # EZKL settings
├── model.compiled              # Compiled circuit
├── pk.key                      # Proving key
├── vk.key                      # Verification key
├── witness.json                # Generated witness
├── proof.json                  # ZK proof
├── <model_name>Verifier.sol    # Solidity contract
├── <model_name>Verifier.json   # Contract ABI
├── calldata.bytes              # Blockchain calldata
├── deploy_<model_name>.js      # Deployment script
└── verify_<model_name>.js      # Verification script
```

## Error Handling

### Common Exceptions

```python
class ModelNotFoundError(Exception):
    """Raised when model name not found in registry"""
    pass

class PipelineError(Exception):
    """Raised when pipeline step fails"""
    pass

class EZKLError(Exception):
    """Raised when EZKL command fails"""
    pass
```

### Error Patterns

```python
try:
    pipeline = EZKLPipeline(model_interface)
    success = pipeline.run_complete_pipeline()
    if not success:
        print("Pipeline failed")
except ModelNotFoundError:
    print("Model not found")
except PipelineError as e:
    print(f"Pipeline error: {e}")
```

## Integration Examples

### Adding Custom Model

```python
# 1. Define model architecture
class CustomNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 2)
        )
    
    def forward(self, x):
        return self.layers(x)

# 2. Implement ModelInterface
class CustomModel(ModelInterface):
    def get_model(self) -> torch.nn.Module:
        model = CustomNN()
        model.eval()
        return model
    
    def get_sample_input(self) -> torch.Tensor:
        return torch.randn(1, 5)
    
    def get_input_data(self) -> List[List[float]]:
        return [[1.0, 2.0, 3.0, 4.0, 5.0]]
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "custom_model",
            "description": "Custom neural network",
            "architecture": "2-layer MLP",
            "domain": "binary_classification"
        }

# 3. Register and use
from models.model_registry import registry
registry.register("custom_model", CustomModel)

# 4. Run pipeline
python demo.py --model custom_model
```

### Programmatic Usage

```python
from core.ezkl_pipeline import EZKLPipeline
from models.model_registry import registry

# Get model
model = registry.create_model("rwkv_simple")

# Run pipeline
pipeline = EZKLPipeline(model)
success = pipeline.run_complete_pipeline()

if success:
    print("Pipeline completed successfully")
    print(f"Output directory: {pipeline.output_dir}")
```

## Performance Tuning

### Model Optimization

- **Reduce Model Size**: Smaller models = faster proofs
- **Simplify Operations**: Avoid complex operations in EZKL
- **Batch Processing**: Use batch size 1 for optimal performance

### EZKL Settings

- **Public I/O**: Faster than private I/O
- **Circuit Size**: Affects proving time and gas costs
- **Key Size**: Balance security vs performance

---

This API reference covers all major components and usage patterns for the Model-Agnostic Pipeline.