#!/usr/bin/env python3
"""
Integration test for EZKL proof generation
Tests all three simple models can generate proofs end-to-end
"""

import subprocess
import json
import time
import os
from pathlib import Path

def run_ezkl_workflow(model_name: str) -> bool:
    """Test complete EZKL workflow for a model"""
    print(f"\n🔍 Testing {model_name}...")
    
    workspace_dir = Path(f"ezkl_workspace/{model_name}")
    if not workspace_dir.exists():
        print(f"❌ Workspace not found: {workspace_dir}")
        return False
    
    original_cwd = os.getcwd()
    os.chdir(workspace_dir)
    
    try:
        # Test witness generation
        print("  1. Generating witness...")
        result = subprocess.run(['ezkl', 'gen-witness', '-D', 'input.json'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"❌ Witness generation failed: {result.stderr}")
            return False
        
        # Test proof generation
        print("  2. Generating proof...")
        start_time = time.time()
        result = subprocess.run(['ezkl', 'prove'], 
                              capture_output=True, text=True, timeout=60)
        proof_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"❌ Proof generation failed: {result.stderr}")
            return False
        
        # Check proof file exists and is valid JSON
        proof_file = Path("proof.json")
        if not proof_file.exists():
            print("❌ Proof file not generated")
            return False
        
        try:
            with open(proof_file, 'r') as f:
                proof_data = json.load(f)
            proof_size = proof_file.stat().st_size
            print(f"✅ Proof generated: {proof_size:,} bytes in {proof_time:.2f}s")
        except json.JSONDecodeError:
            print("❌ Invalid proof JSON")
            return False
        
        # Test proof verification
        print("  3. Verifying proof...")
        result = subprocess.run(['ezkl', 'verify'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            print(f"❌ Proof verification failed: {result.stderr}")
            return False
        
        print(f"✅ {model_name} workflow complete!")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"❌ Timeout during {model_name} workflow")
        return False
    except Exception as e:
        print(f"❌ Error during {model_name} workflow: {e}")
        return False
    finally:
        os.chdir(original_cwd)

def main():
    """Run integration tests for all models"""
    print("🧪 EZKL Integration Test Suite")
    print("=" * 30)
    
    models = ["rwkv_simple", "mamba_simple", "xlstm_simple"]
    results = {}
    
    original_dir = os.getcwd()
    
    try:
        for model in models:
            results[model] = run_ezkl_workflow(model)
        
        # Summary
        print(f"\n📊 Test Results:")
        print("-" * 15)
        
        passed = sum(results.values())
        total = len(results)
        
        for model, success in results.items():
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"  {model}: {status}")
        
        print(f"\n🎯 Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All integration tests passed! The EZKL workflow is working correctly.")
            return True
        else:
            print(f"⚠️  {total - passed} tests failed. Check the errors above.")
            return False
            
    finally:
        os.chdir(original_dir)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)