#!/usr/bin/env python3
"""
Streamlined Model-Agnostic EZKL Demo

Easy model swapping with a single command. Just change the model name to test different architectures.
"""

import sys
import os
from pathlib import Path

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from ezkl_pipeline import EZKLPipeline
from model_registry import get_model, print_models, list_available_models

def run_demo(model_name: str = "weather_xlstm", 
             input_visibility: str = "public",
             output_visibility: str = "public",
             verbose: bool = True):
    """
    Run the complete EZKL demo for any registered model
    
    Args:
        model_name: Name of model to use (see available models below)
        input_visibility: "public" or "private" 
        output_visibility: "public" or "private"
        verbose: Print detailed progress
    """
    
    print("🚀 Model-Agnostic EZKL Demo")
    print("=" * 50)
    
    if verbose:
        print("\n🔧 Available Models:")
        print_models()
    
    try:
        # Create model instance
        print(f"\n📊 Creating model: {model_name}")
        model = get_model(model_name)
        
        # Show model config
        config = model.get_config()
        print(f"   Description: {config.get('description')}")
        print(f"   Architecture: {config.get('architecture')}")
        print(f"   Domain: {config.get('domain')}")
        
        # Run EZKL pipeline
        print(f"\n🔄 Running EZKL pipeline...")
        pipeline = EZKLPipeline(model, work_dir=f"output_{model_name}")
        
        results = pipeline.generate_proof_and_verifier(
            input_visibility=input_visibility,
            output_visibility=output_visibility,
            verbose=verbose
        )
        
        print(f"\n✅ SUCCESS: Complete EZKL workflow for {model_name}")
        print(f"📁 Results saved in: {results['work_dir']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        return None

def run_comparison(models: list = None):
    """Run the demo for multiple models to compare results"""
    
    if models is None:
        models = ["weather_xlstm", "simple_mlp"]
    
    print("🔍 Model Comparison Demo")
    print("=" * 50)
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*20} {model_name.upper()} {'='*20}")
        
        try:
            result = run_demo(model_name, verbose=False)
            if result:
                results[model_name] = result
                print(f"✅ {model_name}: SUCCESS")
            else:
                print(f"❌ {model_name}: FAILED")
        
        except Exception as e:
            print(f"❌ {model_name}: ERROR - {e}")
    
    # Print comparison summary
    print(f"\n📊 Comparison Summary:")
    print("-" * 50)
    
    for model_name, result in results.items():
        proof_size = Path(result['proof_path']).stat().st_size if Path(result['proof_path']).exists() else 0
        contract_size = Path(result['verifier_contract']).stat().st_size if Path(result['verifier_contract']).exists() else 0
        
        print(f"{model_name}:")
        print(f"   Proof size: {proof_size:,} bytes")
        print(f"   Contract size: {contract_size:,} bytes")
        print(f"   Files: {result['work_dir']}")
        print()
    
    return results

def create_blockchain_scripts(model_name: str, results: dict):
    """Create deployment and verification scripts for blockchain testing"""
    
    work_dir = Path(results['work_dir'])
    
    # Create deployment script
    deploy_script = f'''
const hre = require("hardhat");
const fs = require("fs");

async function main() {{
    console.log("🚀 Deploying {model_name} Verifier Contract...");
    
    // Get contract factory
    const Verifier = await hre.ethers.getContractFactory("contracts/{model_name}Verifier.sol:Halo2Verifier");
    
    console.log("⏳ Deploying contract...");
    const verifier = await Verifier.deploy();
    await verifier.waitForDeployment();
    
    const address = await verifier.getAddress();
    console.log("✅ {model_name}Verifier deployed to:", address);
    
    // Save deployment info
    const deploymentInfo = {{
        address: address,
        model: "{model_name}",
        network: hre.network.name,
        timestamp: new Date().toISOString()
    }};
    
    fs.writeFileSync("deployment_{model_name}.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("📁 Deployment info saved");
    
    return address;
}}

main()
    .then(() => process.exit(0))
    .catch((error) => {{
        console.error("💥 Deployment failed:", error);
        process.exit(1);
    }});
'''
    
    # Create verification script
    verify_script = f'''
const hre = require("hardhat");
const fs = require("fs");

async function main() {{
    console.log("🧪 Testing {model_name} Blockchain Verification...");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("deployment_{model_name}.json", "utf8"));
    console.log("📍 Verifier contract:", deploymentInfo.address);
    
    // Read calldata
    const calldataBytes = fs.readFileSync("{results['work_dir']}/calldata.bytes");
    const calldata = "0x" + calldataBytes.toString("hex");
    
    console.log("📦 Calldata length:", calldata.length);
    
    try {{
        const [signer] = await hre.ethers.getSigners();
        
        const gasEstimate = await signer.estimateGas({{
            to: deploymentInfo.address,
            data: calldata
        }});
        console.log("⛽ Gas estimate:", gasEstimate.toString());
        
        const tx = await signer.sendTransaction({{
            to: deploymentInfo.address,
            data: calldata,
            gasLimit: gasEstimate
        }});
        
        const receipt = await tx.wait();
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        if (receipt.status === 1) {{
            console.log("🎉 {model_name} proof verified on blockchain!");
        }}
        
        return receipt.status === 1;
        
    }} catch (error) {{
        console.error("💥 Verification failed:", error.message);
        return false;
    }}
}}

main()
    .then((success) => process.exit(success ? 0 : 1))
    .catch((error) => {{
        console.error("💥 Error:", error);
        process.exit(1);
    }});
'''
    
    # Save scripts
    (work_dir / f"deploy_{model_name}.js").write_text(deploy_script)
    (work_dir / f"verify_{model_name}.js").write_text(verify_script)
    
    print(f"📝 Blockchain scripts created:")
    print(f"   Deploy: {work_dir}/deploy_{model_name}.js")
    print(f"   Verify: {work_dir}/verify_{model_name}.js")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Model-Agnostic EZKL Demo")
    parser.add_argument("--model", default="weather_xlstm", 
                       help=f"Model to use. Available: {list_available_models()}")
    parser.add_argument("--input-visibility", default="public", choices=["public", "private"])
    parser.add_argument("--output-visibility", default="public", choices=["public", "private"])
    parser.add_argument("--compare", action="store_true", help="Run comparison across multiple models")
    parser.add_argument("--blockchain-scripts", action="store_true", help="Generate blockchain deployment scripts")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        print_models()
        sys.exit(0)
    
    if args.compare:
        results = run_comparison()
    else:
        results = run_demo(
            model_name=args.model,
            input_visibility=args.input_visibility,
            output_visibility=args.output_visibility
        )
        
        if results and args.blockchain_scripts:
            create_blockchain_scripts(args.model, results)
    
    print("\n🎯 Demo completed!")