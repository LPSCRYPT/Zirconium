# 🚀 Proof Composition Pipeline - Complete Running Guide

This guide provides step-by-step instructions for running the sequential proof composition pipeline from start to finish.

## 📋 Prerequisites

### Required Software
- **Python 3.8+** with PyTorch, EZKL dependencies
- **Node.js 18+** with npm
- **Hardhat** for smart contract deployment
- **Git** for version control

### Directory Structure
```
proof-composition-pipeline/
├── core/
│   ├── interfaces/          # Model interfaces
│   ├── pipeline/            # Main pipeline engines
│   ├── patterns/            # Advanced composition patterns
│   ├── sequential/          # Sequential execution system
│   └── aggregation/         # Proof aggregation tools
├── chains/                  # Pre-built model chains
├── demo_sequential_composition.py  # Main demo script
└── outputs/                 # Generated artifacts
```

## 🔧 Environment Setup

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install torch ezkl numpy

# Install Node.js dependencies (in main project directory)
npm install

# Verify EZKL installation
ezkl --version
```

### 2. Start Local Blockchain
```bash
# Start Hardhat network (keep running in separate terminal)
npx hardhat node --port 8545
```

## 🏗️ Step 1: Generate Sequential Composition

### Run the Main Demo
```bash
cd /path/to/proof-composition-pipeline

# Generate simple sequential chain (recommended for first run)
python demo_sequential_composition.py --chain-type simple

# Alternative chain types:
python demo_sequential_composition.py --chain-type weather
python demo_sequential_composition.py --chain-type complex
```

### Expected Output
```
⛓️  Sequential Proof Composition Pipeline Demo
============================================================
Mode: Sequential On-Chain Execution
Key Features:
  • Each step feeds output to next step's input
  • True sequential verification on-chain
  • Intermediate values visible and debuggable
  • Atomic execution (all steps succeed or fail)
============================================================

⛓️  Creating Simple Sequential Processing Chain
   Feature Extraction -> Classification -> Decision Making
   Mode: Sequential On-Chain Execution
✅ Simple sequential chain created and validated
   Chain length: 3 models
   Step 0: feature_extractor ([10] → [8])
   Step 1: classifier ([8] → [5])
   Step 2: decision_maker ([5] → [3])

🚀 Starting Sequential Proof Composition
============================================================
⛓️  Starting Sequential Proof Composition Pipeline
   Chain ID: simple_sequential_processing_chain
   Models: 3 models
   Output: sequential_composition_output_simple_sequential_processing_chain
   Mode: Sequential On-Chain Execution
============================================================
```

### What Happens During Generation

1. **Individual Proof Generation** (~5-10 minutes)
   - Each model exported to ONNX format
   - EZKL circuits compiled for each model
   - Zero-knowledge proofs generated
   - Verifier contracts created

2. **Data Flow Execution**
   - Models run sequentially with real data
   - Intermediate values captured and validated
   - Input/output compatibility verified

3. **Contract Generation**
   - Sequential executor contract created
   - Enhanced verifier wrappers generated
   - Deployment and test scripts created

### Generated Files
```
sequential_composition_output_simple_sequential_processing_chain/
├── SequentialChainExecutor.sol          # Main orchestrator
├── EnhancedFeature_ExtractorVerifier.sol # Enhanced verifiers
├── EnhancedClassifierVerifier.sol
├── EnhancedDecision_MakerVerifier.sol
├── step00_feature_extractorVerifier.sol  # EZKL verifiers
├── step01_classifierVerifier.sol
├── step02_decision_makerVerifier.sol
├── deploy_sequential_chain.js            # Deployment script
├── test_sequential_chain_working.js      # Test script
├── README.md                            # Generated documentation
├── sequential_composition.json          # Metadata
└── step*_calldata.bytes                 # Proof calldata
```

## 🚀 Step 2: Deploy to Blockchain

### Copy Contracts
```bash
# Copy generated contracts to main contracts directory
cp sequential_composition_output_simple_sequential_processing_chain/*.sol contracts/

# Verify contracts copied
ls contracts/Sequential* contracts/Enhanced* contracts/step*
```

### Compile Contracts
```bash
# Compile all contracts
npx hardhat compile

# Expected output:
# Compiled 7 Solidity files successfully (evm target: paris).
```

### Deploy Sequential Chain
```bash
cd sequential_composition_output_simple_sequential_processing_chain

# Deploy the complete sequential chain
npx hardhat run deploy_sequential_chain.js --network localhost
```

### Expected Deployment Output
```
🚀 Deploying Sequential Proof Composition Chain
================================================================
⏰ Start time: 2025-07-10T10:19:28.188Z
👤 Deploying with account: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
💰 Account balance: 10000.0 ETH
🌐 Network: localhost | Chain ID: 31337n
================================================================

📋 Deploying underlying EZKL verifiers...
Deploying step 0 underlying verifier: feature_extractor
✅ Step 0 underlying verifier deployed: 0x5FbDB2315678afecb367f032d93F642f64180aa3
Deploying step 1 underlying verifier: classifier
✅ Step 1 underlying verifier deployed: 0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512
Deploying step 2 underlying verifier: decision_maker
✅ Step 2 underlying verifier deployed: 0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0

🔧 Deploying enhanced verifiers...
Deploying enhanced verifier for feature_extractor
✅ Enhanced feature_extractor verifier deployed: 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9
Deploying enhanced verifier for classifier
✅ Enhanced classifier verifier deployed: 0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9
Deploying enhanced verifier for decision_maker
✅ Enhanced decision_maker verifier deployed: 0x5FC8d32690cc91D4c39d9d3abcBD16989F875707

⛓️  Deploying sequential chain executor...
✅ Sequential chain executor deployed: 0x0165878A594ca255338adfa4d48449f69242Eb8F

🔗 Configuring sequential chain...
✅ Chain configured successfully

📁 Deployment info saved to sequential_deployment.json

🎉 SEQUENTIAL DEPLOYMENT SUMMARY
================================================================
⛓️  Sequential Executor: 0x0165878A594ca255338adfa4d48449f69242Eb8F
📊 Enhanced Verifiers:
   0: feature_extractor -> 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9
   1: classifier -> 0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9
   2: decision_maker -> 0x5FC8d32690cc91D4c39d9d3abcBD16989F875707
⏰ Completion time: 2025-07-10T10:19:28.393Z
🎯 Ready for sequential chain execution!
================================================================
```

## 🧪 Step 3: Test Sequential Execution

### Run the Test Suite
```bash
# Test the deployed sequential chain
npx hardhat run test_sequential_chain_working.js --network localhost
```

### Expected Test Output
```
🧪 Testing Sequential Proof Composition Chain
================================================================
📍 Sequential executor: 0x0165878A594ca255338adfa4d48449f69242Eb8F

📋 Chain Information:
   Steps: 3
   Step Names: feature_extractor → classifier → decision_maker

🔧 Testing Individual Enhanced Verifiers...
   ✅ Loaded calldata for step 0: feature_extractor (3780 bytes)
   ✅ Loaded calldata for step 1: classifier (5700 bytes)
   ✅ Loaded calldata for step 2: decision_maker (4036 bytes)

⚡ Testing Sequential Chain Execution...
   Initial input: [359, 118, 517, 312, 845...] (10 values)
   Proof data count: 3
   Estimating gas...
   Estimated gas: 2053959
   Executing sequential chain...
   Waiting for transaction confirmation...

📊 Transaction Results:
📋 Transaction hash: 0xd9fef977e12e2c03501bb4e1e2bce2784ff020f933d553d708d85ff717320bcb
⛽ Gas used: 2053959
✅ Status: SUCCESS

📊 Execution Events:
   Step 0: feature_extractor - ✅
      Output: [0, 0, 0...] (8 values)
      Gas: 503142
   Step 1: classifier - ✅
      Output: [0, 0, 0...] (5 values)
      Gas: 711446
   Step 2: decision_maker - ✅
      Output: [0, 0, 0...] (3 values)
      Gas: 550861

🎉 Chain completed: ✅ SUCCESS
   Final output: [0, 0, 0...] (3 values)
   Total gas: 1816837

📈 Execution Summary:
   Total steps attempted: 3
   Successful steps: 3
   Failed steps: 0
   Overall success: ✅

🔧 Chain Configuration Verification:
   Step 0: feature_extractor
      Verifier: 0xCf7Ed3AccA5a467e9e704C703E8D87F634fB0Fc9
      Input size: 10
      Output size: 8
   Step 1: classifier
      Verifier: 0xDc64a140Aa3E981100a9becA4E685f962f0cF6C9
      Input size: 8
      Output size: 5
   Step 2: decision_maker
      Verifier: 0x5FC8d32690cc91D4c39d9d3abcBD16989F875707
      Input size: 5
      Output size: 3

✅ Sequential chain testing completed!
```

## 📊 Understanding the Results

### Gas Usage Breakdown
- **Total Transaction Gas**: ~2M gas
- **Step 0 (Feature Extractor)**: 503,142 gas
- **Step 1 (Classifier)**: 711,446 gas  
- **Step 2 (Decision Maker)**: 550,861 gas

### Data Flow Verification
```
Input: [10 features] 
  ↓ Step 0: feature_extractor
Output: [8 features] → Input to Step 1
  ↓ Step 1: classifier  
Output: [5 features] → Input to Step 2
  ↓ Step 2: decision_maker
Final Output: [3 features]
```

### What Each Output Means
- **✅ SUCCESS**: All ZK proofs verified correctly
- **Real Transaction**: Actual blockchain state changes
- **Sequential Flow**: Each step's output becomes next step's input
- **Atomic Execution**: All steps succeed or entire chain fails

## 🔧 Customization and Advanced Usage

### Creating Custom Model Chains

1. **Define Your Models**
```python
class MyCustomModel(ComposableModelInterface):
    def get_input_shape(self) -> List[int]:
        return [your_input_size]
    
    def get_output_shape(self) -> List[int]:
        return [your_output_size]
    
    def transform_input(self, data: List[List[float]]) -> List[List[float]]:
        # Your input preprocessing
        return processed_data
    
    def transform_output(self, data: List[List[float]]) -> List[List[float]]:
        # Your output postprocessing
        return processed_data
```

2. **Build Custom Chain**
```python
chain = CompositionChain("my_chain", "My custom processing chain")
chain.add_model(MyModel1())
chain.add_model(MyModel2()) 
chain.add_model(MyModel3())
chain.finalize()
```

### Advanced Composition Patterns

The system also supports:
- **Parallel Execution**: Multiple models process same input
- **Conditional Routing**: Dynamic model selection based on data
- **Tree Compositions**: Branching and merging workflows
- **Loop Compositions**: Iterative processing

See `core/patterns/advanced_compositions.py` for examples.

## 🐛 Troubleshooting

### Common Issues

1. **"Artifact not found" errors**
   ```bash
   # Solution: Recompile contracts
   npx hardhat clean
   npx hardhat compile
   ```

2. **"Network not found" errors**
   ```bash
   # Solution: Ensure Hardhat network is running
   npx hardhat node --port 8545
   ```

3. **EZKL compilation failures**
   ```bash
   # Solution: Check EZKL installation
   ezkl --version
   pip install --upgrade ezkl
   ```

4. **Gas estimation failures**
   ```bash
   # Solution: Increase gas limit in test scripts
   { gasLimit: 5000000 }
   ```

### Debugging Sequential Execution

1. **Check Individual Verifiers**
   ```javascript
   const verifier = await ethers.getContractAt("EnhancedFeature_ExtractorVerifier", address);
   const result = await verifier.verifyAndExecute(proofData, publicInputs);
   ```

2. **Monitor Events**
   ```javascript
   const filter = sequentialExecutor.filters.StepExecuted();
   const events = await sequentialExecutor.queryFilter(filter);
   ```

3. **Validate Chain Configuration**
   ```javascript
   const stepInfo = await sequentialExecutor.getStepInfo(0);
   console.log("Step 0 config:", stepInfo);
   ```

## 🚀 Production Deployment

### Layer 2 Networks
For production use with larger models:
```javascript
// Update hardhat.config.js
networks: {
  polygon: {
    url: "https://polygon-rpc.com/",
    accounts: [process.env.PRIVATE_KEY]
  },
  arbitrum: {
    url: "https://arb1.arbitrum.io/rpc",
    accounts: [process.env.PRIVATE_KEY]
  }
}
```

### Real Model Integration
1. Replace demo models with your trained models
2. Use real data instead of random test inputs
3. Implement proper input/output transformations
4. Add comprehensive error handling

## 📈 Performance Optimization

### Gas Optimization
- Use proof aggregation for chains >5 models
- Implement batch verification for parallel models
- Consider recursive aggregation for very large chains

### Scalability
- Deploy on Layer 2 for lower costs
- Use IPFS for large proof data storage
- Implement caching for repeated computations

---

## 🎯 Next Steps

You now have a complete sequential proof composition system! The pipeline successfully:

✅ **Generates real ZK proofs** for each model  
✅ **Deploys on blockchain** with full transaction logging  
✅ **Executes sequentially** with intermediate value passing  
✅ **Provides transparency** for debugging and verification  
✅ **Scales to complex workflows** with multiple composition patterns  

Ready for production use with your own models and real data!