# ⛓️ Sequential Proof Composition Chain: simple_sequential_processing_chain

**Sequential on-chain: feature extraction -> classification -> decision making**

This directory contains a complete **sequential proof composition system** that executes 3 models on-chain in sequence, where each step's output becomes the next step's input.

## 🏗️ Sequential Architecture

```
feature_extractor → classifier → decision_maker
```

**Key Feature**: True sequential on-chain execution where:
1. Model A verifies its proof AND outputs intermediate values
2. Model B takes A's output as input, verifies its proof, outputs its values  
3. Model C takes B's output as input, verifies its proof, produces final output
4. All steps execute in a single transaction with intermediate values visible on-chain

## 📊 Chain Details

| Step | Model | Input → Output Shape | Domain |
|------|-------|---------------------|---------|
| 0 | `feature_extractor` | [10] → [8] | feature_extraction |\n| 1 | `classifier` | [8] → [5] | classification |\n| 2 | `decision_maker` | [5] → [3] | decision_making |\n

## 🚀 Quick Start

### Deploy the Sequential Chain
```bash
# Copy contracts to main contracts directory
cp *.sol /path/to/contracts/

# Compile all contracts  
npx hardhat compile

# Deploy sequential chain
npx hardhat run deploy_sequential_chain.js --network localhost
```

### Test Sequential Execution
```bash
npx hardhat run test_sequential_chain.js --network localhost
```

## 📁 Contract Architecture

### Core Contracts
- `SequentialChainExecutor.sol` - Main orchestrator for sequential execution
- `Enhanced[ModelName]Verifier.sol` - Wrapper verifiers that return outputs
- `step[N]_[model_name]Verifier.sol` - Underlying EZKL verifiers

### Execution Flow
```solidity
function executeSequentialChain(
    uint256[] memory initialInput,
    bytes[] memory proofData
) external returns (ExecutionResult memory)
```

1. **Step 0**: Verify proof for Model A with `initialInput`, get `outputA`
2. **Step 1**: Verify proof for Model B with `outputA`, get `outputB`  
3. **Step 2**: Verify proof for Model C with `outputB`, get `finalOutput`
4. **Result**: Return complete execution result with all intermediate values

## 🔄 Data Flow

The chain processes data through the following transformations:

```
Initial Input (10 features)
```

↓ Model 0: feature_extractor (8 features)
↓ Model 1: classifier (5 features)
↓ Model 2: decision_maker (3 features)

## ⛽ Gas Considerations

Sequential execution uses more gas than batch verification but provides:
- **Transparency**: All intermediate values visible on-chain
- **Composability**: Easy to add/remove steps or branch execution
- **Debugging**: Clear visibility into which step failed and why
- **Atomicity**: Either entire chain succeeds or fails (no partial states)

## 🎯 Verification Process

1. **Individual Verification**: Each enhanced verifier calls underlying EZKL verifier
2. **Output Extraction**: Intermediate values extracted from public inputs
3. **Sequential Flow**: Output of step N becomes input of step N+1
4. **Final Result**: Complete chain verified with full data lineage

## 🛠️ Development

### Adding Models
Models must implement `ComposableModelInterface` with proper input/output shapes:

```python
class MyModel(ComposableModelInterface):
    def get_input_shape(self) -> List[int]:
        return [N]  # Must match previous model's output shape
        
    def get_output_shape(self) -> List[int]:
        return [M]  # Must match next model's input shape
```

### Testing Individual Steps
```bash
# Test individual enhanced verifier
npx hardhat console --network localhost
> const verifier = await ethers.getContractAt("EnhancedFeatureExtractorVerifier", "0x...");
> const result = await verifier.verifyAndExecute(proofData, publicInputs);
```

### Monitoring Execution
Events are emitted for each step:
```solidity
event StepExecuted(uint256 indexed stepIndex, string stepName, bool success, uint256[] output, uint256 gasUsed);
event ChainCompleted(bool success, uint256 totalSteps, uint256[] finalOutput, uint256 totalGasUsed);
```

## 🔍 Debugging

If execution fails:
1. Check individual step events to identify failing step
2. Verify input/output shape compatibility
3. Ensure proof data matches expected format
4. Test individual verifiers separately

## 📈 Performance

- **Models**: 3
- **Expected Gas**: ~500K base + ~200K per step
- **Intermediate Values**: Visible on-chain for debugging
- **Scalability**: Limited by block gas limit, use Layer 2 for larger chains

---

Generated by Sequential Proof Composition Pipeline
