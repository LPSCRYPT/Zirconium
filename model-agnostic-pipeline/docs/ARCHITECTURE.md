# 🏗️ MAP Architecture Guide

## System Overview

The Model-Agnostic Pipeline (MAP) is a modular system designed for converting any PyTorch model into a blockchain-verifiable zero-knowledge proof system. The architecture follows clean separation of concerns and interface-based design.

## Core Principles

1. **Model Agnostic**: Works with any PyTorch model through a standard interface
2. **Modular Design**: Clear separation between models, pipeline, and infrastructure
3. **Interface-Based**: All models implement a common `ModelInterface`
4. **Pipeline-Driven**: Standardized 10-step EZKL workflow
5. **Developer-Friendly**: Easy to extend and customize

## Directory Structure

```
map/
├── core/                    # 🧠 Core Engine
│   └── ezkl_pipeline.py     # Main pipeline implementation
├── models/                  # 🤖 Model Implementations  
│   ├── model_registry.py    # Factory and registry
│   ├── weather_xlstm.py     # Weather prediction (proven)
│   ├── simple_mlp.py        # Simple classifier
│   ├── rwkv_simple.py       # RWKV language model
│   ├── mamba_simple.py      # Mamba state space model
│   └── xlstm_simple.py      # Simplified xLSTM
├── outputs/                 # 📊 Generated Artifacts
│   ├── output_model1/       # Per-model output directories
│   └── deployment_*.json    # Deployment records
├── docs/                    # 📚 Documentation
│   ├── API.md               # API reference
│   ├── ARCHITECTURE.md      # This file
│   └── CUSTOM_MODELS.md     # How to add models
├── examples/                # 🎯 Example Implementations
├── tools/                   # 🛠️ Utilities
├── tests/                   # 🧪 Test Suite
└── demo.py                  # 🎮 Main CLI Interface
```

## Core Components

### 1. ModelInterface (Abstract Base Class)

```python
class ModelInterface(ABC):
    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return PyTorch model instance"""
    
    @abstractmethod 
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input for ONNX export"""
    
    @abstractmethod
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL JSON format"""
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model metadata"""
```

**Purpose**: Provides a standardized interface that all models must implement, enabling the pipeline to work with any model without knowing its specific implementation details.

### 2. EZKLPipeline (Core Engine)

```python
class EZKLPipeline:
    def __init__(self, model_interface: ModelInterface):
        self.model_interface = model_interface
    
    def run_complete_pipeline(self) -> bool:
        """Execute full 10-step EZKL workflow"""
```

**Purpose**: Orchestrates the complete EZKL workflow from ONNX export to blockchain calldata generation. Works with any model implementing `ModelInterface`.

**10-Step Workflow**:
1. Export model to ONNX
2. Create input data JSON
3. Generate EZKL settings
4. Compile arithmetic circuit
5. Setup proving/verification keys
6. Generate witness from inference
7. Generate zero-knowledge proof
8. Verify proof locally
9. Create Solidity verifier contract
10. Generate blockchain calldata

### 3. ModelRegistry (Factory Pattern)

```python
class ModelRegistry:
    def register(self, name: str, model_class: Type[ModelInterface]):
        """Register new model class"""
    
    def create_model(self, name: str) -> ModelInterface:
        """Create model instance by name"""
```

**Purpose**: Manages available models using the factory pattern, enabling easy model switching and registration of new models.

### 4. Demo CLI (User Interface)

```python
# Command line interface
python demo.py --model <model_name>
python demo.py --list
python demo.py --compare
```

**Purpose**: Provides a user-friendly command-line interface for all MAP operations.

## Data Flow

```
1. User selects model via CLI
         ↓
2. ModelRegistry creates model instance
         ↓
3. EZKLPipeline receives ModelInterface
         ↓
4. Pipeline executes 10-step workflow
         ↓
5. Generated artifacts saved to outputs/
         ↓
6. Optional blockchain deployment scripts
```

## Model Implementation Pattern

### Step 1: Define PyTorch Model

```python
class YourModelArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        # Define layers
        
    def forward(self, x):
        # Forward pass
        return x
```

### Step 2: Implement ModelInterface

```python
class YourModel(ModelInterface):
    def get_model(self) -> torch.nn.Module:
        model = YourModelArchitecture()
        model.eval()
        return model
    
    # Implement other required methods...
```

### Step 3: Register Model

```python
from models.model_registry import registry
registry.register("your_model", YourModel)
```

## Extension Points

### Adding New Pipeline Steps

Extend `EZKLPipeline` class:

```python
class ExtendedEZKLPipeline(EZKLPipeline):
    def run_complete_pipeline(self) -> bool:
        success = super().run_complete_pipeline()
        if success:
            self.your_custom_step()
        return success
    
    def your_custom_step(self):
        # Custom implementation
        pass
```

### Adding Model Capabilities

Extend `ModelInterface`:

```python
class ExtendedModelInterface(ModelInterface):
    @abstractmethod
    def get_preprocessing_steps(self) -> List[str]:
        """Return preprocessing requirements"""
        pass
```

### Custom Output Formats

Override pipeline methods:

```python
def generate_custom_output(self):
    # Custom artifact generation
    pass
```

## Performance Considerations

### Model Size Impact
- **Smaller models**: Faster proof generation, lower gas costs
- **Complex models**: Longer proving times, higher gas consumption
- **Recommended**: Keep models under 100K parameters for optimal performance

### EZKL Settings Optimization
- **Public I/O**: Faster than private I/O
- **Circuit size**: Directly affects proving time
- **Batch size**: Use 1 for optimal performance

### Memory Usage
- **ONNX Export**: ~2x model size
- **Circuit Compilation**: ~5-10x model size  
- **Proof Generation**: ~3-5x circuit size

## Security Considerations

### Input Validation
- All inputs are validated before EZKL processing
- Model outputs are checked for expected ranges
- ONNX exports are verified for correctness

### Private Data Handling
- Models can specify input/output visibility
- Private parameters always remain private
- Proof generation happens in isolated environment

### Contract Security
- Generated verifier contracts use standard Halo2 implementation
- No custom cryptographic code in contracts
- All proofs verified against canonical verification key

## Testing Strategy

### Unit Tests
- Each model implementation
- Pipeline steps individually
- Registry functionality

### Integration Tests
- Full pipeline for each model
- Blockchain deployment and verification
- Cross-model compatibility

### Performance Tests
- Gas consumption measurement
- Proof generation timing
- Memory usage profiling

## Deployment Architecture

### Development
```
Local Machine → MAP → Hardhat Localhost → Verification
```

### Testnet
```
Local Machine → MAP → Testnet (Sepolia/Flow) → Public Verification
```

### Production
```
CI/CD → MAP → Mainnet → Production Verification
```

## Error Handling

### Pipeline Errors
- Each step validates prerequisites
- Graceful failure with detailed error messages
- Automatic cleanup of partial artifacts

### Model Errors
- Input validation before processing
- Model architecture validation
- ONNX export verification

### Blockchain Errors
- Gas estimation before deployment
- Transaction failure handling
- Contract verification validation

## Monitoring and Observability

### Metrics Tracked
- Pipeline execution time per step
- Model-specific performance characteristics
- Gas consumption patterns
- Success/failure rates

### Logging
- Structured logging for all operations
- Pipeline step timing and status
- Error context and stack traces

---

This architecture enables the MAP to be highly modular, extensible, and maintainable while providing a consistent interface for any PyTorch model.