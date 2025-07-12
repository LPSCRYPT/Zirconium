#!/usr/bin/env python3
"""
Sequential Proof Composition Pipeline

Enhanced pipeline for true sequential on-chain verification where each step
feeds its output to the next step's input, with full on-chain execution.
"""

import os
import sys
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional

# Add interfaces to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interfaces'))
from composable_model import ComposableModelInterface, CompositionChain, ChainPosition

# Add MAP core to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'model-agnostic-pipeline', 'core'))
from ezkl_pipeline import EZKLPipeline

# Add sequential executor
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'sequential'))
from sequential_chain_executor import SequentialChainExecutor

class SequentialProofCompositionPipeline:
    """
    Enhanced pipeline for sequential on-chain proof composition.
    
    Key differences from original pipeline:
    1. Each verifier returns outputs that feed into next verifier
    2. True sequential execution on-chain with intermediate values
    3. Enhanced verifiers that wrap EZKL verifiers with output extraction
    4. Single SequentialChainExecutor orchestrates the entire flow
    """
    
    def __init__(self, 
                 chain: CompositionChain,
                 output_dir: str = None,
                 temp_dir: str = None):
        """Initialize the sequential composition pipeline"""
        
        if not chain.is_valid():
            raise ValueError("Chain must be finalized and valid before processing")
        
        self.chain = chain
        self.output_dir = output_dir or f"sequential_composition_output_{chain.chain_id}"
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix=f"sequential_temp_{chain.chain_id}_")
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Track individual model pipelines and proofs
        self.individual_pipelines: List[EZKLPipeline] = []
        self.proofs: List[Dict[str, Any]] = []
        self.data_flow: List[List[List[float]]] = []
        
        # Sequential execution components
        self.sequential_executor = SequentialChainExecutor(chain)
        
    def run_complete_sequential_composition(self) -> bool:
        """Execute the complete sequential composition workflow"""
        
        print(f"⛓️  Starting Sequential Proof Composition Pipeline")
        print(f"   Chain ID: {self.chain.chain_id}")
        print(f"   Models: {len(self.chain)} models")
        print(f"   Output: {self.output_dir}")
        print(f"   Mode: Sequential On-Chain Execution")
        print("=" * 60)
        
        try:
            # Step 1: Generate individual EZKL proofs
            if not self._generate_individual_proofs():
                return False
            
            # Step 2: Execute data flow to capture intermediate values
            if not self._execute_data_flow():
                return False
            
            # Step 3: Generate sequential contracts
            if not self._generate_sequential_contracts():
                return False
            
            # Step 4: Create enhanced verifiers
            if not self._create_enhanced_verifiers():
                return False
            
            # Step 5: Generate deployment and test scripts
            if not self._generate_deployment_artifacts():
                return False
            
            print(f"✅ Sequential composition completed successfully!")
            print(f"📁 Final artifacts in: {self.output_dir}")
            
            return True
            
        except Exception as e:
            print(f"❌ Sequential composition failed: {e}")
            return False
    
    def _generate_individual_proofs(self) -> bool:
        """Generate EZKL proofs for each model in the chain"""
        print("\\n🔧 Step 1: Generating Individual EZKL Proofs")
        print("-" * 40)
        
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            
            print(f"📊 Processing model {i+1}/{len(self.chain)}: {model_name}")
            
            # Create individual model output directory
            model_output_dir = os.path.join(self.temp_dir, f"model_{i}_{model_name}")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Initialize MAP pipeline for this model
            pipeline = EZKLPipeline(
                model=model,
                work_dir=model_output_dir
            )
            
            # Run the complete EZKL workflow
            results = pipeline.generate_proof_and_verifier()
            
            # Check if essential files were generated
            success = (results.get("proof_path") and 
                      results.get("verifier_contract") and
                      os.path.exists(results.get("proof_path", "")))
            
            if not success:
                print(f"❌ Failed to generate proof for {model_name}")
                return False
            
            self.individual_pipelines.append(pipeline)
            
            # Load and store the proof with enhanced metadata
            proof_path = results.get("proof_path")
            with open(proof_path, 'r') as f:
                proof_data = json.load(f)
            
            self.proofs.append({
                "model_name": model_name,
                "position": i,
                "proof_data": proof_data,
                "proof_path": proof_path,
                "output_dir": model_output_dir,
                "input_shape": model.get_input_shape(),
                "output_shape": model.get_output_shape(),
                "verifier_contract_path": results.get("verifier_contract")
            })
            
            print(f"   ✅ Proof generated for {model_name}")
        
        print(f"✅ All {len(self.chain)} individual proofs generated")
        return True
    
    def _execute_data_flow(self) -> bool:
        """Execute the data flow through the model chain"""
        print("\\n🔄 Step 2: Executing Sequential Data Flow")
        print("-" * 40)
        
        # Start with input data from the first model
        current_data = self.chain[0].model.get_input_data()
        self.data_flow.append(current_data)
        
        print(f"🎯 Initial input: {len(current_data[0])} features")
        
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            
            print(f"⛓️  Processing sequential step {i+1}: {model_name}")
            
            # Get the model's PyTorch implementation
            torch_model = model.get_model()
            
            # Convert data to tensor and run inference
            import torch
            input_tensor = torch.tensor(current_data, dtype=torch.float32)
            
            with torch.no_grad():
                output_tensor = torch_model(input_tensor)
                raw_output = output_tensor.numpy().tolist()
            
            print(f"   Input shape: {input_tensor.shape}")
            print(f"   Output shape: {output_tensor.shape}")
            
            # Transform output for next model (if not the last model)
            if i < len(self.chain) - 1:
                transformed_output = model.transform_output(raw_output)
                next_model = self.chain[i + 1].model
                current_data = next_model.transform_input(transformed_output)
                
                self.data_flow.append(current_data)
                print(f"   Transformed for next model: {len(current_data[0])} features")
            else:
                # Last model - store final output
                final_output = model.transform_output(raw_output)
                self.data_flow.append(final_output)
                print(f"   Final output: {len(final_output[0])} features")
        
        print(f"✅ Sequential data flow completed through {len(self.chain)} models")
        return True
    
    def _generate_sequential_contracts(self) -> bool:
        """Generate contracts for sequential execution"""
        print("\\n🏗️ Step 3: Generating Sequential Contracts")
        print("-" * 40)
        
        # Generate all sequential contracts
        contracts = self.sequential_executor.generate_sequential_contracts(self.output_dir)
        
        for contract_name, contract_code in contracts.items():
            contract_path = os.path.join(self.output_dir, contract_name)
            with open(contract_path, 'w') as f:
                f.write(contract_code)
            print(f"📋 Generated {contract_name}")
        
        print(f"✅ Generated {len(contracts)} sequential contracts")
        return True
    
    def _create_enhanced_verifiers(self) -> bool:
        """Copy and organize verifier contracts"""
        print("\\n🔧 Step 4: Creating Enhanced Verifier System")
        print("-" * 40)
        
        # Copy individual EZKL verifiers to output directory
        for i, proof_info in enumerate(self.proofs):
            model_name = proof_info["model_name"]
            source_dir = proof_info["output_dir"]
            
            # Find the EZKL verifier contract
            verifier_sol = None
            for file in os.listdir(source_dir):
                if file.endswith("Verifier.sol"):
                    verifier_sol = file
                    break
            
            if verifier_sol:
                # Copy to output directory with step prefix
                source_path = os.path.join(source_dir, verifier_sol)
                dest_path = os.path.join(self.output_dir, f"step{i:02d}_{model_name}Verifier.sol")
                shutil.copy2(source_path, dest_path)
                
                print(f"   📋 Copied EZKL verifier for {model_name}")
                
                # Also copy calldata for testing
                calldata_file = os.path.join(source_dir, "calldata.bytes")
                if os.path.exists(calldata_file):
                    dest_calldata = os.path.join(self.output_dir, f"step{i:02d}_{model_name}_calldata.bytes")
                    shutil.copy2(calldata_file, dest_calldata)
        
        print(f"✅ Enhanced verifier system created")
        return True
    
    def _generate_deployment_artifacts(self) -> bool:
        """Generate deployment and test scripts"""
        print("\\n📦 Step 5: Generating Deployment Artifacts")
        print("-" * 40)
        
        # Generate deployment script
        deployment_script = self.sequential_executor.generate_sequential_deployment_script(self.output_dir)
        deployment_path = os.path.join(self.output_dir, "deploy_sequential_chain.js")
        with open(deployment_path, 'w') as f:
            f.write(deployment_script)
        print(f"🚀 Generated deployment script")
        
        # Generate test script
        test_script = self.sequential_executor.generate_sequential_test_script()
        test_path = os.path.join(self.output_dir, "test_sequential_chain.js")
        with open(test_path, 'w') as f:
            f.write(test_script)
        print(f"🧪 Generated test script")
        
        # Generate comprehensive README
        readme = self._generate_sequential_readme()
        readme_path = os.path.join(self.output_dir, "README.md")
        with open(readme_path, 'w') as f:
            f.write(readme)
        print(f"📋 Generated README.md")
        
        # Generate composition metadata
        metadata = self._generate_sequential_metadata()
        metadata_path = os.path.join(self.output_dir, "sequential_composition.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📊 Generated composition metadata")
        
        print(f"✅ All deployment artifacts generated")
        return True
    
    def _generate_sequential_readme(self) -> str:
        """Generate comprehensive README for sequential composition"""
        
        models_table = ""
        for i, position in enumerate(self.chain):
            model = position.model
            config = model.get_config()
            input_shape = model.get_input_shape()
            output_shape = model.get_output_shape()
            models_table += f"| {i} | `{config['name']}` | {input_shape} → {output_shape} | {config.get('domain', 'general')} |\\n"
        
        readme = f"""# ⛓️ Sequential Proof Composition Chain: {self.chain.chain_id}

**{self.chain.description}**

This directory contains a complete **sequential proof composition system** that executes {len(self.chain)} models on-chain in sequence, where each step's output becomes the next step's input.

## 🏗️ Sequential Architecture

```
{' → '.join([pos.model.get_config()['name'] for pos in self.chain])}
```

**Key Feature**: True sequential on-chain execution where:
1. Model A verifies its proof AND outputs intermediate values
2. Model B takes A's output as input, verifies its proof, outputs its values  
3. Model C takes B's output as input, verifies its proof, produces final output
4. All steps execute in a single transaction with intermediate values visible on-chain

## 📊 Chain Details

| Step | Model | Input → Output Shape | Domain |
|------|-------|---------------------|---------|
{models_table}

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
Initial Input ({len(self.data_flow[0][0]) if self.data_flow else 0} features)
```

{chr(10).join([f"↓ Model {i}: {self.chain[i].model.get_config()['name']}" + (f" ({len(self.data_flow[i+1][0]) if i+1 < len(self.data_flow) else 0} features)" if i+1 < len(self.data_flow) else " (final output)") for i in range(len(self.chain))])}

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

- **Models**: {len(self.chain)}
- **Expected Gas**: ~500K base + ~200K per step
- **Intermediate Values**: Visible on-chain for debugging
- **Scalability**: Limited by block gas limit, use Layer 2 for larger chains

---

Generated by Sequential Proof Composition Pipeline
"""
        
        return readme
    
    def _generate_sequential_metadata(self) -> Dict[str, Any]:
        """Generate metadata for sequential composition"""
        
        return {
            "chain_id": self.chain.chain_id,
            "description": self.chain.description,
            "composition_type": "sequential_on_chain",
            "models": [
                {
                    "position": i,
                    "name": pos.model.get_config()['name'],
                    "input_shape": pos.model.get_input_shape(),
                    "output_shape": pos.model.get_output_shape(),
                    "domain": pos.model.get_config().get('domain', 'general')
                }
                for i, pos in enumerate(self.chain)
            ],
            "data_flow_shapes": [
                len(step[0]) if step else 0 for step in self.data_flow
            ],
            "execution_model": {
                "type": "sequential",
                "on_chain": True,
                "intermediate_values_visible": True,
                "atomic_execution": True
            },
            "generated_contracts": [
                "SequentialChainExecutor.sol",
                *[f"Enhanced{pos.model.get_config()['name'].title()}Verifier.sol" for pos in self.chain],
                *[f"step{i:02d}_{pos.model.get_config()['name']}Verifier.sol" for i, pos in enumerate(self.chain)]
            ],
            "deployment_scripts": [
                "deploy_sequential_chain.js",
                "test_sequential_chain.js"
            ],
            "timestamp": json.dumps({}),  # Will be filled with actual timestamp
        }
    
    def cleanup(self):
        """Clean up temporary directories"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Cleaned up temporary directory: {self.temp_dir}")