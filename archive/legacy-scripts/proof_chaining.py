#!/usr/bin/env python3
"""
Proof-Chaining Workflow for RWKV → Mamba → xLSTM
Demonstrates sequential zkML model composition with cryptographic proofs
"""

import json
import subprocess
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil

class ProofChainWorkflow:
    """Orchestrates proof-chaining across RWKV → Mamba → xLSTM models"""
    
    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or os.getcwd())
        self.workspace_dir = self.base_dir / "ezkl_workspace"
        self.models = ["rwkv_simple", "mamba_simple", "xlstm_simple"]
        self.chain_results = {}
        
        # Verify all models exist
        for model in self.models:
            model_dir = self.workspace_dir / model
            if not model_dir.exists():
                raise FileNotFoundError(f"Model workspace not found: {model_dir}")
    
    def extract_model_outputs(self, model_name: str) -> List[float]:
        """Extract rescaled outputs from a model's witness file with validation"""
        witness_file = self.workspace_dir / model_name / "witness.json"
        
        if not witness_file.exists():
            raise FileNotFoundError(f"Witness file not found: {witness_file}")
        
        # Validate file is readable and not empty
        try:
            if witness_file.stat().st_size == 0:
                raise ValueError(f"Witness file is empty: {witness_file}")
        except OSError as e:
            raise FileNotFoundError(f"Cannot access witness file: {e}")
        
        # Load and validate JSON structure
        try:
            with open(witness_file, 'r', encoding='utf-8') as f:
                witness_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in witness file: {e}")
        except UnicodeDecodeError as e:
            raise ValueError(f"Invalid file encoding in witness file: {e}")
        except OSError as e:
            raise FileNotFoundError(f"Cannot read witness file: {e}")
        
        # Validate witness data structure
        if not isinstance(witness_data, dict):
            raise ValueError("Witness data must be a JSON object")
        
        if "pretty_elements" not in witness_data:
            raise ValueError("Missing 'pretty_elements' in witness data")
        
        pretty_elements = witness_data["pretty_elements"]
        if not isinstance(pretty_elements, dict):
            raise ValueError("'pretty_elements' must be an object")
        
        if "rescaled_outputs" not in pretty_elements:
            raise ValueError("Missing 'rescaled_outputs' in witness data")
        
        rescaled_outputs = pretty_elements["rescaled_outputs"]
        if not isinstance(rescaled_outputs, list) or len(rescaled_outputs) == 0:
            raise ValueError("'rescaled_outputs' must be a non-empty array")
        
        # Extract first instance outputs
        output_values = rescaled_outputs[0]
        if not isinstance(output_values, list):
            raise ValueError("Output values must be an array")
        
        # Convert string values to floats with validation
        try:
            float_outputs = []
            for i, val in enumerate(output_values):
                if isinstance(val, str):
                    float_val = float(val)
                elif isinstance(val, (int, float)):
                    float_val = float(val)
                else:
                    raise ValueError(f"Invalid output value type at index {i}: {type(val)}")
                
                # Basic sanity check for reasonable values
                if abs(float_val) > 1e10:
                    raise ValueError(f"Output value seems unreasonable: {float_val}")
                
                float_outputs.append(float_val)
            
            return float_outputs
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Failed to convert output values to floats: {e}")
    
    def create_chained_input(self, outputs: List[float], model_name: str) -> str:
        """Create input.json for the next model in the chain with validation"""
        
        # Validate inputs
        if not outputs:
            raise ValueError("No outputs provided for chaining")
        
        if len(outputs) < 10:
            raise ValueError(f"Insufficient outputs for chaining: got {len(outputs)}, need at least 10")
        
        if not model_name:
            raise ValueError("Model name cannot be empty")
        
        # Validate model directory exists
        model_dir = self.workspace_dir / model_name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Take first 10 values for next model input
        chained_input = outputs[:10]
        
        # Validate output values are reasonable
        for i, val in enumerate(chained_input):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Invalid output type at index {i}: {type(val)}")
            if not (-1e6 < val < 1e6):  # Reasonable range check
                raise ValueError(f"Output value out of reasonable range at index {i}: {val}")
        
        # Create input JSON in the correct format
        input_data = {
            "input_data": [chained_input]
        }
        
        # Write to the model's input.json with error handling
        input_file = model_dir / "input.json"
        
        try:
            # Create backup if file exists
            if input_file.exists():
                backup_file = model_dir / "input.json.backup"
                try:
                    shutil.copy2(input_file, backup_file)
                except OSError:
                    pass  # Continue even if backup fails
            
            # Write new input file
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(input_data, f, indent=2, ensure_ascii=False)
            
            # Verify the file was written correctly
            if not input_file.exists():
                raise OSError("Input file was not created")
            
            # Verify file content by reading it back
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    verification_data = json.load(f)
                    
                if verification_data != input_data:
                    raise ValueError("File content verification failed")
                    
            except (json.JSONDecodeError, OSError) as e:
                raise ValueError(f"Failed to verify written file: {e}")
        
        except OSError as e:
            raise RuntimeError(f"Failed to write input file: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error writing input file: {e}")
        
        print(f"   📄 Created chained input for {model_name}: {len(chained_input)} values")
        return str(input_file)
    
    def run_ezkl_proof(self, model_name: str) -> Dict:
        """Run complete EZKL workflow for a model with comprehensive error handling"""
        
        print(f"\n🔗 Processing {model_name}...")
        
        model_dir = self.workspace_dir / model_name
        original_cwd = os.getcwd()
        
        # Validate prerequisites
        try:
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            required_files = ["model.onnx", "settings.json", "pk.key", "vk.key"]
            missing_files = [f for f in required_files if not (model_dir / f).exists()]
            if missing_files:
                raise FileNotFoundError(f"Missing required files: {missing_files}")
            
            # Check input file exists and is valid
            input_file = model_dir / "input.json"
            if not input_file.exists():
                raise FileNotFoundError(f"Input file not found: {input_file}")
            
            # Validate input JSON format
            try:
                with open(input_file, 'r') as f:
                    input_data = json.load(f)
                if "input_data" not in input_data or not isinstance(input_data["input_data"], list):
                    raise ValueError("Invalid input format: must contain 'input_data' array")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in input file: {e}")
            
            os.chdir(model_dir)
            
            # Step 1: Generate witness with retry logic
            print("   1. Generating witness...")
            start_time = time.time()
            
            for attempt in range(3):  # Try up to 3 times
                try:
                    result = subprocess.run(['ezkl', 'gen-witness', '-D', 'input.json'], 
                                          capture_output=True, text=True, timeout=45)
                    if result.returncode == 0:
                        break
                    elif attempt == 2:  # Last attempt
                        error_msg = result.stderr.strip() if result.stderr else "Unknown witness generation error"
                        if "timeout" in error_msg.lower():
                            raise TimeoutError(f"Witness generation timed out after 45s")
                        elif "memory" in error_msg.lower() or "allocation" in error_msg.lower():
                            raise MemoryError(f"Out of memory during witness generation: {error_msg}")
                        elif "circuit" in error_msg.lower():
                            raise ValueError(f"Circuit compilation issue: {error_msg}")
                        else:
                            raise RuntimeError(f"Witness generation failed: {error_msg}")
                    else:
                        print(f"   ⚠️ Witness attempt {attempt + 1} failed, retrying...")
                        time.sleep(1)
                except subprocess.TimeoutExpired:
                    if attempt == 2:
                        raise TimeoutError("Witness generation timed out after 45s")
                    print(f"   ⚠️ Witness timeout attempt {attempt + 1}, retrying...")
                    time.sleep(2)
            
            witness_time = time.time() - start_time
            
            # Verify witness file was created
            witness_file = model_dir / "witness.json"
            if not witness_file.exists():
                raise FileNotFoundError("Witness file was not generated")
            
            # Step 2: Generate proof with enhanced error handling
            print("   2. Generating proof...")
            start_time = time.time()
            
            for attempt in range(2):  # Try up to 2 times
                try:
                    result = subprocess.run(['ezkl', 'prove'], 
                                          capture_output=True, text=True, timeout=120)
                    if result.returncode == 0:
                        break
                    elif attempt == 1:  # Last attempt
                        error_msg = result.stderr.strip() if result.stderr else "Unknown proof generation error"
                        if "timeout" in error_msg.lower():
                            raise TimeoutError(f"Proof generation timed out after 120s")
                        elif "memory" in error_msg.lower():
                            raise MemoryError(f"Out of memory during proof generation: {error_msg}")
                        elif "key" in error_msg.lower():
                            raise ValueError(f"Proving key issue: {error_msg}")
                        else:
                            raise RuntimeError(f"Proof generation failed: {error_msg}")
                    else:
                        print(f"   ⚠️ Proof attempt {attempt + 1} failed, retrying...")
                        time.sleep(2)
                except subprocess.TimeoutExpired:
                    if attempt == 1:
                        raise TimeoutError("Proof generation timed out after 120s")
                    print(f"   ⚠️ Proof timeout attempt {attempt + 1}, retrying...")
                    time.sleep(5)
            
            proof_time = time.time() - start_time
            
            # Verify proof file was created and has reasonable size
            proof_file = model_dir / "proof.json"
            if not proof_file.exists():
                raise FileNotFoundError("Proof file was not generated")
            
            proof_size = proof_file.stat().st_size
            if proof_size < 1000:  # Suspiciously small proof
                raise ValueError(f"Generated proof is too small ({proof_size} bytes)")
            elif proof_size > 100_000_000:  # Suspiciously large proof (100MB)
                raise ValueError(f"Generated proof is too large ({proof_size} bytes)")
            
            # Step 3: Verify proof
            print("   3. Verifying proof...")
            try:
                result = subprocess.run(['ezkl', 'verify'], 
                                      capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    error_msg = result.stderr.strip() if result.stderr else "Unknown verification error"
                    raise RuntimeError(f"Proof verification failed: {error_msg}")
            except subprocess.TimeoutExpired:
                raise TimeoutError("Proof verification timed out after 60s")
            
            # Step 4: Generate EVM calldata with error handling
            print("   4. Generating EVM calldata...")
            try:
                result = subprocess.run(['ezkl', 'encode-evm-calldata', 
                                       '--proof-path=proof.json', '--calldata-path=calldata.txt'], 
                                      capture_output=True, text=True, timeout=30)
                if result.returncode != 0:
                    print(f"   ⚠️ EVM calldata generation failed: {result.stderr}")
                    calldata_success = False
                else:
                    # Verify calldata file was created
                    calldata_file = model_dir / "calldata.txt"
                    if calldata_file.exists() and calldata_file.stat().st_size > 0:
                        print("   ✅ EVM calldata generated")
                        calldata_success = True
                    else:
                        print("   ⚠️ EVM calldata file is empty or missing")
                        calldata_success = False
            except subprocess.TimeoutExpired:
                print("   ⚠️ EVM calldata generation timed out")
                calldata_success = False
            
            # Step 5: Extract outputs for chaining with validation
            try:
                outputs = self.extract_model_outputs(model_name)
                if not outputs or len(outputs) < 10:
                    raise ValueError(f"Insufficient model outputs: got {len(outputs)}, need at least 10")
            except Exception as e:
                raise RuntimeError(f"Failed to extract model outputs: {e}")
            
            print(f"   ✅ Proof complete: {proof_size:,} bytes in {proof_time:.2f}s")
            print(f"   📊 Output: {len(outputs)} values")
            
            return {
                "model": model_name,
                "proof_file": str(proof_file),
                "proof_size": proof_size,
                "proof_time": proof_time,
                "witness_time": witness_time,
                "outputs": outputs,
                "calldata_success": calldata_success,
                "success": True
            }
            
        except (FileNotFoundError, ValueError, RuntimeError, MemoryError, TimeoutError) as e:
            error_type = type(e).__name__
            print(f"   ❌ {error_type}: {e}")
            return {
                "model": model_name,
                "error": str(e),
                "error_type": error_type,
                "success": False
            }
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            return {
                "model": model_name,
                "error": f"Unexpected error: {e}",
                "error_type": "UnexpectedError",
                "success": False
            }
        finally:
            os.chdir(original_cwd)
    
    def run_proof_chain(self, initial_input: List[float] = None) -> Dict:
        """Execute the complete proof-chaining workflow"""
        
        print("🔗 Starting Proof-Chaining Workflow")
        print("=" * 40)
        print("Pipeline: RWKV → Mamba → xLSTM")
        
        # Use default input if none provided
        if initial_input is None:
            initial_input = [
                0.4652051031589508, -0.9299561381340027, -1.242055892944336,
                1.0063905715942383, 1.0123469829559326, -0.12968644499778748,
                -0.7979132533073425, -0.5822405815124512, 2.378202199935913,
                1.0034712553024292
            ]
        
        print(f"📊 Initial input: {len(initial_input)} values")
        
        chain_results = {
            "pipeline": ["rwkv_simple", "mamba_simple", "xlstm_simple"],
            "initial_input": initial_input,
            "steps": [],
            "proofs": [],
            "total_time": 0,
            "success": True
        }
        
        current_input = initial_input
        total_start = time.time()
        
        # Step 1: RWKV (uses initial input)
        print(f"\n🚀 Step 1: RWKV Processing")
        
        # Create input for RWKV
        self.create_chained_input(current_input, "rwkv_simple")
        
        # Run RWKV proof generation
        rwkv_result = self.run_ezkl_proof("rwkv_simple")
        chain_results["steps"].append(rwkv_result)
        
        if not rwkv_result["success"]:
            chain_results["success"] = False
            return chain_results
        
        chain_results["proofs"].append(rwkv_result["proof_file"])
        
        # Step 2: Mamba (uses RWKV outputs)
        print(f"\n🚀 Step 2: Mamba Processing")
        
        # Chain RWKV outputs to Mamba
        self.create_chained_input(rwkv_result["outputs"], "mamba_simple")
        
        # Run Mamba proof generation
        mamba_result = self.run_ezkl_proof("mamba_simple")
        chain_results["steps"].append(mamba_result)
        
        if not mamba_result["success"]:
            chain_results["success"] = False
            return chain_results
        
        chain_results["proofs"].append(mamba_result["proof_file"])
        
        # Step 3: xLSTM (uses Mamba outputs)
        print(f"\n🚀 Step 3: xLSTM Processing")
        
        # Chain Mamba outputs to xLSTM
        self.create_chained_input(mamba_result["outputs"], "xlstm_simple")
        
        # Run xLSTM proof generation
        xlstm_result = self.run_ezkl_proof("xlstm_simple")
        chain_results["steps"].append(xlstm_result)
        
        if not xlstm_result["success"]:
            chain_results["success"] = False
            return chain_results
        
        chain_results["proofs"].append(xlstm_result["proof_file"])
        
        # Final results
        chain_results["total_time"] = time.time() - total_start
        chain_results["final_output"] = xlstm_result["outputs"]
        
        return chain_results
    
    def display_chain_results(self, results: Dict):
        """Display detailed results of the proof-chaining workflow"""
        
        print(f"\n🎯 Proof-Chaining Results")
        print("=" * 25)
        
        if not results["success"]:
            print("❌ Chain failed - check individual step errors above")
            return
        
        print(f"✅ Successfully chained {len(results['pipeline'])} models")
        print(f"⏱️  Total time: {results['total_time']:.2f}s")
        print(f"📊 Final output: {len(results['final_output'])} values")
        
        print(f"\n📋 Step-by-step breakdown:")
        for i, step in enumerate(results["steps"]):
            model = step["model"]
            proof_size = step["proof_size"]
            proof_time = step["proof_time"]
            print(f"   {i+1}. {model}: {proof_size:,} bytes in {proof_time:.2f}s")
        
        print(f"\n🔐 Generated proofs:")
        for i, proof_file in enumerate(results["proofs"]):
            print(f"   {i+1}. {proof_file}")
        
        print(f"\n💡 Proof Chain Composition:")
        print(f"   Input({len(results['initial_input'])}) → RWKV → Mamba → xLSTM → Output({len(results['final_output'])})")
        
        # Show data flow
        print(f"\n🔄 Data Flow:")
        print(f"   Initial: {results['initial_input'][:3]}... ({len(results['initial_input'])} values)")
        
        for i, step in enumerate(results["steps"]):
            if step["success"]:
                outputs = step["outputs"]
                print(f"   {step['model']}: {outputs[:3]}... ({len(outputs)} values)")
        
        print(f"\n🎉 Proof-chaining workflow completed successfully!")
        print(f"🔗 Three cryptographic proofs demonstrate verifiable model composition")

def main():
    """Run the proof-chaining demonstration"""
    
    try:
        # Initialize workflow
        workflow = ProofChainWorkflow()
        
        # Run the complete chain
        results = workflow.run_proof_chain()
        
        # Display results
        workflow.display_chain_results(results)
        
        # Save results to file
        results_file = "data/proof_chain_results.json"
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {results_file}")
        
        return results["success"]
        
    except Exception as e:
        print(f"❌ Proof-chaining failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)