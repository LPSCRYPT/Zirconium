#!/usr/bin/env python3
"""
Test EVM verifier on fresh localhost blockchain.
Following EZKL demo with public inputs and outputs.
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
    print("🚀 Localhost Blockchain Test (Public Inputs & Outputs)")
    print("Following EZKL demo pattern with blockchain deployment")
    
    # Create working directory
    work_dir = Path("/Users/bot/code/zirconium/localhost_test")
    work_dir.mkdir(exist_ok=True)
    os.chdir(work_dir)
    
    # Step 1: Create model and export to ONNX
    print("\n📊 Step 1: Creating Model with Public I/O")
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
    
    # Step 3: Generate settings with PUBLIC visibility
    print("\n⚙️ Step 3: Generating Settings (Public I/O)")
    result = subprocess.run([
        'ezkl', 'gen-settings', '-M', 'model.onnx', 
        '--input-visibility', 'public', '--output-visibility', 'public'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Settings generation failed: {result.stderr}")
        return False
    print("✅ Settings generated with public I/O")
    
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
    
    # Step 7: Generate proof
    print("\n🔒 Step 7: Generating Proof")
    result = subprocess.run([
        'ezkl', 'prove', '-M', 'model.compiled', '-W', 'witness.json', 
        '--pk-path', 'pk.key', '--proof-path', 'proof.json'
    ], capture_output=True, text=True, timeout=300)
    
    if result.returncode != 0:
        print(f"❌ Proof generation failed: {result.stderr}")
        return False
    print("✅ Proof generated")
    
    # Step 8: Verify proof locally
    print("\n✅ Step 8: Verifying Proof Locally")
    result = subprocess.run([
        'ezkl', 'verify', '--proof-path', 'proof.json', '--vk-path', 'vk.key', '-S', 'settings.json'
    ], capture_output=True, text=True, timeout=60)
    
    if result.returncode != 0:
        print(f"❌ Local proof verification failed: {result.stderr}")
        return False
    print("✅ Local verification successful!")
    
    # Step 9: Create EVM verifier contract
    print("\n🏗️ Step 9: Creating EVM Verifier Contract")
    result = subprocess.run([
        'ezkl', 'create-evm-verifier', '--vk-path', 'vk.key', 
        '--sol-code-path', 'WeatherVerifier.sol', '--abi-path', 'WeatherVerifier.json', 
        '--settings-path', 'settings.json'
    ], capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        print(f"❌ EVM verifier creation failed: {result.stderr}")
        return False
    print("✅ EVM verifier contract created!")
    
    # Step 10: Create deployment script
    print("\n📝 Step 10: Creating Deployment Script")
    
    deployment_script = '''
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🚀 Deploying Weather Verifier Contract...");
    
    // Read the contract source
    const contractSource = fs.readFileSync("WeatherVerifier.sol", "utf8");
    
    // Get contract factory
    const WeatherVerifier = await hre.ethers.getContractFactory("Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await WeatherVerifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ WeatherVerifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {
        address: address,
        network: hre.network.name,
        timestamp: new Date().toISOString()
    };
    
    fs.writeFileSync("deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved to deployment.json");
    
    return address;
}

main()
    .then((address) => {
        console.log("🎉 Deployment successful!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    });
'''
    
    with open("deploy.js", "w") as f:
        f.write(deployment_script)
    print("✅ Deployment script created")
    
    # Step 11: Create test script
    print("\n🧪 Step 11: Creating Blockchain Test Script")
    
    test_script = '''
const hre = require("hardhat");
const fs = require("fs");

async function main() {
    console.log("🧪 Testing Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Load proof data
    const proofData = JSON.parse(fs.readFileSync("proof.json", "utf8"));
    console.log("🔐 Proof loaded, size:", JSON.stringify(proofData).length, "chars");
    
    // Get contract instance
    const WeatherVerifier = await hre.ethers.getContractFactory("Halo2Verifier");
    const verifier = WeatherVerifier.attach(deploymentInfo.address);
    
    // Extract public instances from proof
    const publicInstances = proofData.instances[0];
    console.log("📊 Public instances:", publicInstances.length);
    console.log("🔢 Values:", publicInstances.slice(0, 4).join(", "));
    
    // Convert proof to bytes
    const proofBytes = "0x" + Buffer.from(proofData.proof).toString("hex");
    console.log("📦 Proof bytes length:", proofBytes.length);
    
    try {
        console.log("⏳ Calling verifyProof on blockchain...");
        
        // Call verify function
        const result = await verifier.verifyProof(proofBytes, publicInstances);
        
        if (result) {
            console.log("🎉 SUCCESS: Blockchain verification passed!");
            console.log("✅ Proof verified on-chain");
        } else {
            console.log("❌ FAILED: Blockchain verification rejected");
        }
        
        return result;
        
    } catch (error) {
        console.error("💥 Blockchain verification error:", error.message);
        return false;
    }
}

main()
    .then((success) => {
        if (success) {
            console.log("🎯 Test completed successfully!");
        } else {
            console.log("💥 Test failed");
        }
        process.exit(success ? 0 : 1);
    })
    .catch((error) => {
        console.error("💥 Test error:", error);
        process.exit(1);
    });
'''
    
    with open("test_blockchain.js", "w") as f:
        f.write(test_script)
    print("✅ Blockchain test script created")
    
    # Display summary
    print("\n📊 Ready for Blockchain Testing:")
    
    try:
        # Check proof
        with open("proof.json", "r") as f:
            proof_data = json.load(f)
        
        proof_size = Path("proof.json").stat().st_size
        public_instances = len(proof_data.get('instances', [[]])[0])
        
        print(f"   Proof size: {proof_size:,} bytes")
        print(f"   Public instances: {public_instances}")
        print(f"   Contract: WeatherVerifier.sol ({Path('WeatherVerifier.sol').stat().st_size:,} bytes)")
        
        if "pretty_public_inputs" in proof_data:
            inputs = proof_data["pretty_public_inputs"].get("rescaled_inputs", [])
            outputs = proof_data["pretty_public_inputs"].get("rescaled_outputs", [])
            if inputs:
                print(f"   Public inputs: {len(inputs[0])} values")
            if outputs:
                print(f"   Public outputs: {len(outputs[0])} values")
        
    except Exception as e:
        print(f"   Warning: Could not read proof details: {e}")
    
    print(f"\n📁 All files ready in: {work_dir}")
    print("\n🚀 Next steps:")
    print("   1. Start hardhat node: npx hardhat node")
    print("   2. Deploy contract: npx hardhat run deploy.js --network localhost")
    print("   3. Test verification: npx hardhat run test_blockchain.js --network localhost")
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Ready for localhost blockchain testing!")
    else:
        print("\n💥 Setup failed")