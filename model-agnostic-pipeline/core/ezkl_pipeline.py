#!/usr/bin/env python3
"""
Model-Agnostic EZKL Pipeline

This pipeline can work with any PyTorch model that implements the ModelInterface.
Simply swap in different models without changing the EZKL workflow.
"""

import os
import json
import torch
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Tuple
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Interface that all models must implement for EZKL compatibility"""
    
    @abstractmethod
    def get_model(self) -> torch.nn.Module:
        """Return the PyTorch model instance"""
        pass
    
    @abstractmethod
    def get_sample_input(self) -> torch.Tensor:
        """Return sample input tensor for the model"""
        pass
    
    @abstractmethod
    def get_input_data(self) -> List[List[float]]:
        """Return input data in EZKL JSON format"""
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return model configuration (name, description, etc.)"""
        pass

class EZKLPipeline:
    """Model-agnostic EZKL proof generation and verification pipeline"""
    
    def __init__(self, model: ModelInterface, work_dir: str = None):
        self.model = model
        self.work_dir = Path(work_dir) if work_dir else Path("ezkl_workspace")
        self.work_dir.mkdir(exist_ok=True)
        
        # Get model config
        self.config = model.get_config()
        self.model_name = self.config.get("name", "model")
        
        print(f"🚀 Initialized EZKL Pipeline for: {self.model_name}")
        print(f"📁 Working directory: {self.work_dir}")
    
    def generate_proof_and_verifier(self, 
                                   input_visibility: str = "public",
                                   output_visibility: str = "public",
                                   verbose: bool = True) -> Dict[str, Any]:
        """
        Complete EZKL workflow: model → proof → verifier contract
        
        Args:
            input_visibility: "public" or "private"
            output_visibility: "public" or "private" 
            verbose: Print detailed progress
            
        Returns:
            Dict with all generated file paths and metadata
        """
        
        if verbose:
            print(f"\n🔄 Starting EZKL workflow for {self.model_name}")
            print(f"   Input visibility: {input_visibility}")
            print(f"   Output visibility: {output_visibility}")
        
        # Change to working directory
        original_dir = os.getcwd()
        os.chdir(self.work_dir)
        
        try:
            results = {}
            
            # Step 1: Export to ONNX
            if verbose:
                print("\n📊 Step 1: Exporting model to ONNX")
            
            model_instance = self.model.get_model()
            sample_input = self.model.get_sample_input()
            
            model_path = f"{self.model_name}.onnx"
            torch.onnx.export(
                model_instance,
                sample_input,
                model_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            results['onnx_path'] = str(self.work_dir / model_path)
            
            if verbose:
                print(f"   ✅ Model exported: {model_path}")
            
            # Step 2: Create input data
            if verbose:
                print("\n📋 Step 2: Creating input data")
            
            input_data = {"input_data": self.model.get_input_data()}
            input_path = "input.json"
            with open(input_path, "w") as f:
                json.dump(input_data, f)
            results['input_path'] = str(self.work_dir / input_path)
            
            if verbose:
                print(f"   ✅ Input data created: {input_path}")
            
            # Step 3: Generate EZKL settings
            if verbose:
                print("\n⚙️ Step 3: Generating EZKL settings")
            
            cmd = ['ezkl', 'gen-settings', '-M', model_path]
            if input_visibility == "public":
                cmd.extend(['--input-visibility', 'public'])
            if output_visibility == "public":
                cmd.extend(['--output-visibility', 'public'])
                
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise Exception(f"Settings generation failed: {result.stderr}")
            
            results['settings_path'] = str(self.work_dir / "settings.json")
            if verbose:
                print("   ✅ Settings generated")
            
            # Step 4: Compile circuit
            if verbose:
                print("\n🔧 Step 4: Compiling circuit")
            
            result = subprocess.run([
                'ezkl', 'compile-circuit', '-M', model_path, '-S', 'settings.json'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise Exception(f"Circuit compilation failed: {result.stderr}")
            
            results['compiled_path'] = str(self.work_dir / "model.compiled")
            if verbose:
                print("   ✅ Circuit compiled")
            
            # Step 5: Setup keys
            if verbose:
                print("\n🔐 Step 5: Setting up proving/verification keys")
            
            result = subprocess.run([
                'ezkl', 'setup', '-M', 'model.compiled', 
                '--vk-path', 'vk.key', '--pk-path', 'pk.key'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise Exception(f"Key setup failed: {result.stderr}")
            
            results['vk_path'] = str(self.work_dir / "vk.key")
            results['pk_path'] = str(self.work_dir / "pk.key")
            if verbose:
                print("   ✅ Keys generated")
            
            # Step 6: Generate witness
            if verbose:
                print("\n👁️ Step 6: Generating witness")
            
            result = subprocess.run([
                'ezkl', 'gen-witness', '-D', 'input.json', 
                '-M', 'model.compiled', '-O', 'witness.json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise Exception(f"Witness generation failed: {result.stderr}")
            
            results['witness_path'] = str(self.work_dir / "witness.json")
            if verbose:
                print("   ✅ Witness generated")
            
            # Step 7: Generate proof
            if verbose:
                print("\n🔒 Step 7: Generating zero-knowledge proof")
            
            result = subprocess.run([
                'ezkl', 'prove', '-M', 'model.compiled', '-W', 'witness.json',
                '--pk-path', 'pk.key', '--proof-path', 'proof.json'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise Exception(f"Proof generation failed: {result.stderr}")
            
            results['proof_path'] = str(self.work_dir / "proof.json")
            if verbose:
                print("   ✅ Proof generated")
            
            # Step 8: Verify proof locally
            if verbose:
                print("\n✅ Step 8: Verifying proof locally")
            
            result = subprocess.run([
                'ezkl', 'verify', '--proof-path', 'proof.json',
                '--vk-path', 'vk.key', '-S', 'settings.json'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise Exception(f"Local verification failed: {result.stderr}")
            
            if verbose:
                print("   ✅ Local verification successful")
            
            # Step 9: Create EVM verifier contract
            if verbose:
                print("\n🏗️ Step 9: Creating EVM verifier contract")
            
            verifier_name = f"{self.model_name}Verifier"
            result = subprocess.run([
                'ezkl', 'create-evm-verifier', '--vk-path', 'vk.key',
                '--sol-code-path', f'{verifier_name}.sol',
                '--abi-path', f'{verifier_name}.json',
                '--settings-path', 'settings.json'
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise Exception(f"EVM verifier creation failed: {result.stderr}")
            
            results['verifier_contract'] = str(self.work_dir / f"{verifier_name}.sol")
            results['verifier_abi'] = str(self.work_dir / f"{verifier_name}.json")
            if verbose:
                print(f"   ✅ Verifier contract created: {verifier_name}.sol")
            
            # Step 10: Generate blockchain calldata
            if verbose:
                print("\n📞 Step 10: Generating blockchain calldata")
            
            result = subprocess.run([
                'ezkl', 'encode-evm-calldata', '--proof-path', 'proof.json',
                '--calldata-path', 'calldata.bytes'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                if verbose:
                    print("   ⚠️ Calldata generation failed (may not be needed)")
            else:
                results['calldata_path'] = str(self.work_dir / "calldata.bytes")
                if verbose:
                    print("   ✅ Calldata generated")
            
            # Collect metadata
            results.update({
                'model_name': self.model_name,
                'model_config': self.config,
                'work_dir': str(self.work_dir),
                'input_visibility': input_visibility,
                'output_visibility': output_visibility
            })
            
            # Get file sizes for summary
            if verbose:
                self._print_summary(results)
            
            return results
            
        finally:
            os.chdir(original_dir)
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of generated files and their sizes"""
        print(f"\n📊 Pipeline Summary for {self.model_name}:")
        
        file_info = [
            ('ONNX Model', results.get('onnx_path')),
            ('ZK Proof', results.get('proof_path')),
            ('Verifier Contract', results.get('verifier_contract')),
            ('Verifier ABI', results.get('verifier_abi')),
            ('Calldata', results.get('calldata_path'))
        ]
        
        for name, path in file_info:
            if path and Path(path).exists():
                size = Path(path).stat().st_size
                print(f"   {name}: {size:,} bytes")
        
        print(f"📁 All files in: {results['work_dir']}")
        print("✨ Ready for blockchain deployment!")

def run_pipeline_for_model(model: ModelInterface, **kwargs) -> Dict[str, Any]:
    """Convenience function to run the complete pipeline for any model"""
    pipeline = EZKLPipeline(model)
    return pipeline.generate_proof_and_verifier(**kwargs)