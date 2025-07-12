#!/usr/bin/env python3
"""
Aggregated proof workflow for blockchain compatibility.
Based on research notes: use aggregated proofs even for single proofs.
"""

import os
import json
import torch
import torch.nn as nn
import subprocess
from pathlib import Path

# Same model as working demo
class xLSTMCell(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.forget_gate = nn.Linear(hidden_size, hidden_size)
        self.input_gate = nn.Linear(hidden_size, hidden_size)
        self.candidate_gate = nn.Linear(hidden_size, hidden_size)
        self.output_gate = nn.Linear(hidden_size, hidden_size)
        self.exp_gate = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        forget = torch.sigmoid(self.forget_gate(x))
        input_g = torch.sigmoid(self.input_gate(x))
        candidate = torch.tanh(self.candidate_gate(x))
        output = torch.sigmoid(self.output_gate(x))
        exp_factor = torch.exp(self.exp_gate * input_g)
        cell_state = forget * candidate * exp_factor
        hidden = output * torch.tanh(cell_state)
        return self.layer_norm(hidden)

class WeatherxLSTMModel(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 32, output_size: int = 4):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.xlstm1 = xLSTMCell(hidden_size)
        self.xlstm2 = xLSTMCell(hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size)
        self._init_weather_weights()
    
    def _init_weather_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_proj = self.input_proj(x)
        h1 = self.xlstm1(x_proj)
        h2 = self.xlstm2(h1)
        output = self.output_proj(h2)
        return output

def main():
    print("🚀 Aggregated Proof Workflow for Blockchain Compatibility")
    print("Based on research: aggregated proofs solve 'too big for chain' issue")
    
    # Create working directory
    work_dir = Path("/Users/bot/code/zirconium/aggregated_demo")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    # Step 1: Create model and export to ONNX
    print("\n📊 Step 1: Creating and Exporting Model")
    model = WeatherxLSTMModel(input_size=10, hidden_size=32, output_size=4)
    model.eval()
    
    sample_input = torch.tensor([[0.5, 0.6, 0.4, 0.7, 0.5, 0.8, 0.6, 0.4, 0.5, 0.7]], dtype=torch.float32)
    
    torch.onnx.export(
        model,
        sample_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['weather_sequence'],
        output_names=['prediction']
    )
    print("✅ Model exported to ONNX")
    
    # Step 2: Create input JSON
    print("\n📋 Step 2: Creating Input Data")
    input_data = {"input_data": [[0.5, 0.6, 0.4, 0.7, 0.5, 0.8, 0.6, 0.4, 0.5, 0.7]]}
    with open("input.json", "w") as f:
        json.dump(input_data, f)
    print("✅ Input data created")
    
    # Step 3: Generate settings
    print("\n⚙️ Step 3: Generating EZKL Settings")
    result = subprocess.run([
        'ezkl', 'gen-settings', '-M', 'model.onnx'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Settings generation failed: {result.stderr}")
        return False
    print("✅ Settings generated")
    
    # Step 4: Compile circuit
    print("\n🔧 Step 4: Compiling Circuit")
    result = subprocess.run([
        'ezkl', 'compile-circuit', '-M', 'model.onnx', '-S', 'settings.json'
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ Circuit compilation failed: {result.stderr}")
        return False
    print("✅ Circuit compiled")
    
    # Step 5: Setup keys
    print("\n🔐 Step 5: Setting up Keys")
    result = subprocess.run([
        'ezkl', 'setup', '-M', 'model.compiled', '--vk-path', 'vk.key', '--pk-path', 'pk.key'
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ Setup failed: {result.stderr}")
        return False
    print("✅ Keys generated")
    
    # Step 6: Generate witness
    print("\n👁️ Step 6: Generating Witness")
    result = subprocess.run([
        'ezkl', 'gen-witness', '-D', 'input.json', '-M', 'model.compiled', '-O', 'witness.json'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Witness generation failed: {result.stderr}")
        return False
    print("✅ Witness generated")
    
    # Step 7: Generate proof FOR AGGREGATION
    print("\n🔒 Step 7: Generating Proof for Aggregation")
    print("   Key difference: Using 'for-aggr' flag")
    result = subprocess.run([
        'ezkl', 'prove', '-M', 'model.compiled', '-W', 'witness.json', 
        '--pk-path', 'pk.key', '--proof-path', 'proof_for_aggr.json', '--proof-type=for-aggr'
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"❌ Proof generation failed: {result.stderr}")
        return False
    print("✅ Proof for aggregation generated")
    
    # Step 8: Setup aggregation (even for single proof)
    print("\n🔗 Step 8: Setting up Aggregation")
    print("   Key insight: Use aggregation even for single proof")
    result = subprocess.run([
        'ezkl', 'setup-aggregate', '--sample-snarks=proof_for_aggr.json', 
        '--vk-path=vk_aggr.key', '--pk-path=pk_aggr.key', '--logrows=17'
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ Aggregation setup failed: {result.stderr}")
        return False
    print("✅ Aggregation setup complete")
    
    # Step 9: Create aggregated proof
    print("\n🎯 Step 9: Creating Aggregated Proof")
    result = subprocess.run([
        'ezkl', 'aggregate', '--aggregation-snarks', 'proof_for_aggr.json', 
        '--proof-path', 'proof_aggr.json', '--vk-path', 'vk_aggr.key', '--logrows', '17'
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"❌ Aggregation failed: {result.stderr}")
        return False
    print("✅ Aggregated proof created")
    
    # Step 10: Verify aggregated proof
    print("\n✅ Step 10: Verifying Aggregated Proof")
    result = subprocess.run([
        'ezkl', 'verify-aggr', '--proof-path', 'proof_aggr.json', '--vk-path', 'vk_aggr.key', '--logrows', '17'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Aggregated proof verification failed: {result.stderr}")
        return False
    
    print("🎉 SUCCESS: Aggregated proof verified!")
    
    # Step 11: Create EVM verifier contract
    print("\n🏗️ Step 11: Creating EVM Verifier Contract")
    print("   Using create-evm-verifier-aggr for blockchain compatibility")
    result = subprocess.run([
        'ezkl', 'create-evm-verifier-aggr', '--vk-path', 'vk_aggr.key', 
        '--sol-code-path', 'WeatherVerifier.sol', '--abi-path', 'WeatherVerifier.json', '--logrows', '17'
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ EVM verifier creation failed: {result.stderr}")
        return False
    
    print("✅ EVM verifier contract created!")
    
    # Display results
    print("\n📊 Aggregated Proof Summary:")
    
    try:
        # Standard proof size
        standard_size = Path("proof_for_aggr.json").stat().st_size
        print(f"   Standard proof size: {standard_size:,} bytes")
        
        # Aggregated proof size
        aggr_size = Path("proof_aggr.json").stat().st_size
        print(f"   Aggregated proof size: {aggr_size:,} bytes")
        
        # Verifier contract
        if Path("WeatherVerifier.sol").exists():
            verifier_size = Path("WeatherVerifier.sol").stat().st_size
            print(f"   Verifier contract size: {verifier_size:,} bytes")
            print("   ✅ Ready for blockchain deployment!")
        
    except Exception as e:
        print(f"   Warning: Could not read file details: {e}")
    
    print(f"\n📁 All files saved in: {work_dir}")
    print("✨ Aggregated proof workflow completed successfully!")
    print("🎯 Ready for blockchain verification!")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🔗 Aggregated proof approach working!")
        print("🚀 Next: Deploy verifier contract and test blockchain verification")
    else:
        print("\n💥 Aggregated proof workflow failed")