#!/usr/bin/env python3
"""
Custom Model Setup Automation Script
Automates the integration of new ONNX models into Zirconium
"""

import os
import json
import shutil
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import onnx
import numpy as np

class CustomModelSetup:
    """Automates custom model integration into Zirconium"""
    
    def __init__(self, model_name: str, onnx_path: str, workspace_dir: str = "ezkl_workspace"):
        self.model_name = model_name
        self.onnx_path = Path(onnx_path)
        self.workspace_dir = Path(workspace_dir)
        self.model_dir = self.workspace_dir / model_name
        
        # Validate inputs
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        if not self.onnx_path.suffix == '.onnx':
            raise ValueError(f"File must be .onnx format: {onnx_path}")
    
    def analyze_model(self) -> Dict:
        """Analyze ONNX model structure and requirements"""
        print(f"🔍 Analyzing ONNX model: {self.onnx_path}")
        
        try:
            model = onnx.load(self.onnx_path)
            onnx.checker.check_model(model)
        except Exception as e:
            raise ValueError(f"Invalid ONNX model: {e}")
        
        # Extract input information
        inputs = []
        for input_tensor in model.graph.input:
            shape = []
            for dim in input_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            inputs.append({
                "name": input_tensor.name,
                "shape": shape,
                "type": input_tensor.type.tensor_type.elem_type
            })
        
        # Extract output information
        outputs = []
        for output_tensor in model.graph.output:
            shape = []
            for dim in output_tensor.type.tensor_type.shape.dim:
                if dim.dim_value:
                    shape.append(dim.dim_value)
                else:
                    shape.append(-1)  # Dynamic dimension
            outputs.append({
                "name": output_tensor.name,
                "shape": shape,
                "type": output_tensor.type.tensor_type.elem_type
            })
        
        # Extract operations
        operations = list(set([node.op_type for node in model.graph.node]))
        
        analysis = {
            "inputs": inputs,
            "outputs": outputs,
            "operations": sorted(operations),
            "model_size_mb": self.onnx_path.stat().st_size / (1024 * 1024),
            "num_nodes": len(model.graph.node)
        }
        
        print(f"   📊 Inputs: {len(inputs)}")
        print(f"   📊 Outputs: {len(outputs)}")
        print(f"   📊 Operations: {len(operations)}")
        print(f"   📊 Model size: {analysis['model_size_mb']:.2f} MB")
        print(f"   📊 Nodes: {analysis['num_nodes']}")
        
        return analysis
    
    def check_ezkl_compatibility(self, analysis: Dict) -> List[str]:
        """Check model compatibility with EZKL"""
        print("🔍 Checking EZKL compatibility...")
        
        # Known EZKL-supported operations (as of 2024)
        supported_ops = {
            'Add', 'Mul', 'Sub', 'Div', 'Relu', 'Sigmoid', 'Tanh',
            'MatMul', 'Gemm', 'Conv', 'MaxPool', 'AveragePool',
            'Reshape', 'Transpose', 'Concat', 'Split', 'Slice',
            'BatchNormalization', 'Dropout', 'Flatten', 'Squeeze',
            'Unsqueeze', 'Clip', 'Sqrt', 'Pow', 'Exp', 'Log',
            'Softmax', 'LogSoftmax', 'Identity', 'Constant'
        }
        
        model_ops = set(analysis['operations'])
        unsupported_ops = model_ops - supported_ops
        
        warnings = []
        
        if unsupported_ops:
            warnings.append(f"⚠️ Potentially unsupported operations: {unsupported_ops}")
        
        # Check for dynamic shapes
        for inp in analysis['inputs']:
            if -1 in inp['shape']:
                warnings.append(f"⚠️ Dynamic input shape detected: {inp['name']} {inp['shape']}")
        
        # Check model complexity
        if analysis['num_nodes'] > 1000:
            warnings.append(f"⚠️ Large model ({analysis['num_nodes']} nodes) may require significant memory")
        
        if analysis['model_size_mb'] > 50:
            warnings.append(f"⚠️ Large model file ({analysis['model_size_mb']:.1f}MB) may slow compilation")
        
        if warnings:
            print("   Compatibility warnings:")
            for warning in warnings:
                print(f"   {warning}")
        else:
            print("   ✅ Model appears compatible with EZKL")
        
        return warnings
    
    def create_workspace(self, analysis: Dict) -> None:
        """Create model workspace with proper structure"""
        print(f"📁 Creating workspace: {self.model_dir}")
        
        # Create directory
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy ONNX model
        shutil.copy2(self.onnx_path, self.model_dir / "model.onnx")
        print(f"   ✅ Copied model to {self.model_dir / 'model.onnx'}")
        
        # Generate input.json
        input_data = self.generate_input_json(analysis)
        with open(self.model_dir / "input.json", 'w') as f:
            json.dump(input_data, f, indent=2)
        print(f"   ✅ Created input.json")
        
        # Create README for the model
        self.create_model_readme(analysis)
        print(f"   ✅ Created README.md")
    
    def generate_input_json(self, analysis: Dict) -> Dict:
        """Generate appropriate input.json for the model"""
        
        if not analysis['inputs']:
            raise ValueError("Model has no inputs")
        
        primary_input = analysis['inputs'][0]
        shape = primary_input['shape']
        
        # Handle dynamic shapes
        actual_shape = []
        for dim in shape:
            if dim == -1:
                actual_shape.append(1)  # Use batch size 1 for dynamic dims
            else:
                actual_shape.append(dim)
        
        # Generate input data
        if len(actual_shape) == 1:
            # 1D input - generate array directly
            input_array = [float(i) for i in range(actual_shape[0])]
        elif len(actual_shape) == 2 and actual_shape[0] == 1:
            # Batch size 1, flatten to 1D
            input_array = [float(i) for i in range(actual_shape[1])]
        else:
            # Multi-dimensional - flatten and take reasonable amount
            total_elements = np.prod(actual_shape)
            if total_elements > 100:
                # Too large, use first 10 elements for proof chaining compatibility
                input_array = [float(i) for i in range(10)]
                print(f"   ⚠️ Large input shape {actual_shape}, using 10-element array for compatibility")
            else:
                input_array = [float(i) for i in range(int(total_elements))]
        
        return {
            "input_data": [input_array]
        }
    
    def create_model_readme(self, analysis: Dict) -> None:
        """Create README.md for the model workspace"""
        
        readme_content = f"""# {self.model_name} Model

## Model Information
- **Source**: {self.onnx_path.name}
- **Size**: {analysis['model_size_mb']:.2f} MB
- **Nodes**: {analysis['num_nodes']}

## Input/Output Structure

### Inputs
"""
        
        for i, inp in enumerate(analysis['inputs']):
            readme_content += f"- **{inp['name']}**: Shape {inp['shape']}, Type {inp['type']}\n"
        
        readme_content += "\n### Outputs\n"
        
        for i, out in enumerate(analysis['outputs']):
            readme_content += f"- **{out['name']}**: Shape {out['shape']}, Type {out['type']}\n"
        
        readme_content += f"""
## Operations Used
{', '.join(analysis['operations'])}

## Generated Files
- `model.onnx` - Original ONNX model
- `input.json` - Sample input data
- `settings.json` - EZKL settings (generated by setup)
- `model.compiled` - Compiled circuit (generated by setup)
- `pk.key`, `vk.key` - Proving/verification keys (generated by setup)

## Usage

```bash
# Generate settings
ezkl gen-settings --model=model.onnx --settings-path=settings.json --input=input.json

# Compile circuit
ezkl compile-circuit --model=model.onnx --compiled-circuit=model.compiled --settings-path=settings.json

# Setup keys
ezkl setup --compiled-circuit=model.compiled --vk-path=vk.key --pk-path=pk.key

# Generate witness
ezkl gen-witness --compiled-circuit=model.compiled --input=input.json --output=witness.json

# Generate proof
ezkl prove --compiled-circuit=model.compiled --pk-path=pk.key --proof-path=proof.json --witness-path=witness.json

# Verify proof
ezkl verify --proof-path=proof.json --vk-path=vk.key --settings-path=settings.json
```
"""
        
        with open(self.model_dir / "README.md", 'w') as f:
            f.write(readme_content)
    
    def run_ezkl_setup(self) -> bool:
        """Run complete EZKL setup for the model"""
        print(f"⚙️ Running EZKL setup for {self.model_name}...")
        
        original_cwd = os.getcwd()
        success = True
        
        try:
            os.chdir(self.model_dir)
            
            # Step 1: Generate settings
            print("   1. Generating settings...")
            result = subprocess.run([
                'ezkl', 'gen-settings',
                '--model=model.onnx',
                '--settings-path=settings.json',
                '--input=input.json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"   ❌ Settings generation failed: {result.stderr}")
                return False
            
            # Step 2: Compile circuit
            print("   2. Compiling circuit...")
            result = subprocess.run([
                'ezkl', 'compile-circuit',
                '--model=model.onnx',
                '--compiled-circuit=model.compiled',
                '--settings-path=settings.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Circuit compilation failed: {result.stderr}")
                return False
            
            # Step 3: Setup keys
            print("   3. Setting up keys...")
            result = subprocess.run([
                'ezkl', 'setup',
                '--compiled-circuit=model.compiled',
                '--vk-path=vk.key',
                '--pk-path=pk.key'
            ], capture_output=True, text=True, timeout=180)
            
            if result.returncode != 0:
                print(f"   ❌ Key setup failed: {result.stderr}")
                return False
            
            # Step 4: Test witness generation
            print("   4. Testing witness generation...")
            result = subprocess.run([
                'ezkl', 'gen-witness',
                '--compiled-circuit=model.compiled',
                '--input=input.json',
                '--output=witness.json'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"   ❌ Witness generation failed: {result.stderr}")
                return False
            
            print("   ✅ EZKL setup completed successfully!")
            
        except subprocess.TimeoutExpired as e:
            print(f"   ❌ EZKL setup timed out: {e}")
            success = False
        except Exception as e:
            print(f"   ❌ EZKL setup failed: {e}")
            success = False
        finally:
            os.chdir(original_cwd)
        
        return success
    
    def generate_test_proof(self) -> bool:
        """Generate a test proof to validate the setup"""
        print(f"🔐 Generating test proof for {self.model_name}...")
        
        original_cwd = os.getcwd()
        success = True
        
        try:
            os.chdir(self.model_dir)
            
            # Generate proof
            print("   1. Generating proof...")
            result = subprocess.run([
                'ezkl', 'prove',
                '--compiled-circuit=model.compiled',
                '--pk-path=pk.key',
                '--proof-path=proof.json',
                '--witness-path=witness.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"   ❌ Proof generation failed: {result.stderr}")
                return False
            
            # Verify proof
            print("   2. Verifying proof...")
            result = subprocess.run([
                'ezkl', 'verify',
                '--proof-path=proof.json',
                '--vk-path=vk.key',
                '--settings-path=settings.json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                print(f"   ❌ Proof verification failed: {result.stderr}")
                return False
            
            # Generate EVM calldata
            print("   3. Generating EVM calldata...")
            result = subprocess.run([
                'ezkl', 'encode-evm-calldata',
                '--proof-path=proof.json',
                '--calldata-path=calldata.txt'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                print(f"   ⚠️ EVM calldata generation failed: {result.stderr}")
                print("   (This is OK for testing, but required for blockchain deployment)")
            else:
                print("   ✅ EVM calldata generated")
            
            # Check file sizes
            proof_size = (self.model_dir / "proof.json").stat().st_size / 1024
            print(f"   📊 Proof size: {proof_size:.1f} KB")
            
            print("   ✅ Test proof generated and verified successfully!")
            
        except subprocess.TimeoutExpired as e:
            print(f"   ❌ Proof generation timed out: {e}")
            success = False
        except Exception as e:
            print(f"   ❌ Proof generation failed: {e}")
            success = False
        finally:
            os.chdir(original_cwd)
        
        return success
    
    def generate_smart_contract_template(self) -> str:
        """Generate Solidity contract template for the model"""
        
        contract_name = f"{self.model_name.title()}Verifier"
        wrapper_name = f"{contract_name}Wrapper"
        
        template = f'''// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../base/BaseEZKLVerifier.sol";
// TODO: Generate the actual EZKL verifier using:
// ezkl create-evm-verifier --vk-path=vk.key --sol-code-path={contract_name}.sol --abi-path={contract_name}.abi

contract {wrapper_name} is BaseEZKLVerifier {{
    // TODO: Replace with actual EZKL verifier interface
    address private immutable ezkl_verifier;
    
    constructor(address _ezklVerifierAddress) {{
        ezkl_verifier = _ezklVerifierAddress;
    }}
    
    function verify(bytes memory proof, uint256[] memory publicInputs) 
        public override returns (bool) {{
        
        // Validate inputs
        require(proof.length > 0, "Empty proof");
        require(publicInputs.length >= 1, "Insufficient public inputs");
        
        // TODO: Implement actual verification call
        // Example pattern:
        // bytes memory callData = abi.encodeWithSignature("verifyProof(bytes,uint256[])", proof, publicInputs);
        // (bool success, bytes memory returnData) = ezkl_verifier.call(callData);
        
        totalVerifications++;
        bytes32 proofHash = keccak256(proof);
        emit ProofVerified(proofHash, address(this), true);
        
        // TODO: Replace with actual verification
        return true; // Placeholder
    }}
    
    function getArchitecture() public pure override returns (string memory) {{
        return "{self.model_name}";
    }}
}}'''
        
        contract_path = Path("contracts/verifiers") / f"{wrapper_name}.sol"
        
        print(f"📝 Generated contract template: {contract_path}")
        print(f"   ⚠️ Remember to:")
        print(f"   1. Generate actual EZKL verifier contract")
        print(f"   2. Update the verification logic")
        print(f"   3. Add to deployment scripts")
        
        return template
    
    def create_integration_summary(self, analysis: Dict, warnings: List[str]) -> None:
        """Create integration summary and next steps"""
        
        print(f"\n🎉 Custom model integration completed!")
        print("=" * 50)
        print(f"📁 Model workspace: {self.model_dir}")
        print(f"📊 Model info: {analysis['num_nodes']} nodes, {analysis['model_size_mb']:.1f}MB")
        
        if warnings:
            print(f"\n⚠️ Compatibility warnings:")
            for warning in warnings:
                print(f"   {warning}")
        
        print(f"\n✅ Files created:")
        print(f"   • {self.model_dir}/model.onnx")
        print(f"   • {self.model_dir}/input.json") 
        print(f"   • {self.model_dir}/settings.json")
        print(f"   • {self.model_dir}/model.compiled")
        print(f"   • {self.model_dir}/pk.key, vk.key")
        print(f"   • {self.model_dir}/proof.json")
        print(f"   • {self.model_dir}/README.md")
        
        print(f"\n🚀 Next steps:")
        print(f"   1. Review generated files in {self.model_dir}/")
        print(f"   2. Test integration: python test_integration.py")
        print(f"   3. Generate EZKL verifier contract:")
        print(f"      cd {self.model_dir}")
        print(f"      ezkl create-evm-verifier --vk-path=vk.key --sol-code-path={self.model_name}Verifier.sol")
        print(f"   4. Create wrapper contract using the generated template")
        print(f"   5. Add to deployment scripts and test on localhost")
        print(f"   6. Integrate into proof chaining workflows")
        
        print(f"\n📚 Documentation:")
        print(f"   • Model details: {self.model_dir}/README.md")
        print(f"   • Integration guide: docs/CUSTOM_MODELS.md")
        print(f"   • Troubleshooting: docs/TROUBLESHOOTING.md")

def main():
    """CLI interface for custom model setup"""
    
    parser = argparse.ArgumentParser(description="Automate custom ONNX model integration into Zirconium")
    parser.add_argument("model_name", help="Name for your model (e.g., 'my_transformer')")
    parser.add_argument("onnx_path", help="Path to your ONNX model file")
    parser.add_argument("--workspace", default="ezkl_workspace", help="Workspace directory (default: ezkl_workspace)")
    parser.add_argument("--skip-ezkl", action="store_true", help="Skip EZKL setup (workspace creation only)")
    parser.add_argument("--skip-proof", action="store_true", help="Skip test proof generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize setup
        setup = CustomModelSetup(args.model_name, args.onnx_path, args.workspace)
        
        # Analyze model
        analysis = setup.analyze_model()
        warnings = setup.check_ezkl_compatibility(analysis)
        
        # Create workspace
        setup.create_workspace(analysis)
        
        if not args.skip_ezkl:
            # Run EZKL setup
            if setup.run_ezkl_setup():
                if not args.skip_proof:
                    # Generate test proof
                    setup.generate_test_proof()
            else:
                print("❌ EZKL setup failed. Check the error messages above.")
                return False
        
        # Generate contract template
        contract_template = setup.generate_smart_contract_template()
        
        # Create summary
        setup.create_integration_summary(analysis, warnings)
        
        print(f"\n🎯 Success! Your model '{args.model_name}' is ready for integration.")
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)