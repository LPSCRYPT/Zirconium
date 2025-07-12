#!/usr/bin/env python3
"""
Sequential Proof Composition Pipeline Demo

Demonstrates creating sequential on-chain verified chains where each step
feeds its output to the next step's input, with full on-chain execution.
"""

import sys
import os
import argparse

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chains'))

from composable_model import CompositionChain
from sequential_composition_pipeline import SequentialProofCompositionPipeline
from simple_chain_models import (
    ComposableFeatureExtractor,
    ComposableClassifier, 
    ComposableDecisionMaker,
    ComposableWeatherProcessor,
    ComposableSequenceModel
)

def create_simple_sequential_chain() -> CompositionChain:
    """Create a simple 3-model sequential chain: Feature Extraction -> Classification -> Decision"""
    
    print("⛓️  Creating Simple Sequential Processing Chain")
    print("   Feature Extraction -> Classification -> Decision Making")
    print("   Mode: Sequential On-Chain Execution")
    
    chain = CompositionChain(
        chain_id="simple_sequential_processing_chain",
        description="Sequential on-chain: feature extraction -> classification -> decision making"
    )
    
    # Add models in sequence
    chain.add_model(ComposableFeatureExtractor())
    chain.add_model(ComposableClassifier())
    chain.add_model(ComposableDecisionMaker())
    
    # Finalize and validate
    if chain.finalize():
        print("✅ Simple sequential chain created and validated")
        print(f"   Chain length: {len(chain)} models")
        
        # Print data flow information
        for i, position in enumerate(chain):
            model = position.model
            config = model.get_config()
            input_shape = model.get_input_shape()
            output_shape = model.get_output_shape()
            print(f"   Step {i}: {config['name']} ({input_shape} → {output_shape})")
        
        return chain
    else:
        print("❌ Simple sequential chain validation failed")
        return None

def create_weather_sequential_chain() -> CompositionChain:
    """Create a weather processing sequential chain"""
    
    print("⛓️  Creating Weather Sequential Processing Chain")
    print("   Weather Data -> Processing -> Prediction")
    print("   Mode: Sequential On-Chain Execution")
    
    chain = CompositionChain(
        chain_id="weather_sequential_chain",
        description="Sequential weather data processing and prediction"
    )
    
    # Add weather-specific models
    chain.add_model(ComposableWeatherProcessor())
    chain.add_model(ComposableSequenceModel())
    chain.add_model(ComposableDecisionMaker())
    
    if chain.finalize():
        print("✅ Weather sequential chain created and validated")
        return chain
    else:
        print("❌ Weather sequential chain validation failed")
        return None

def create_complex_sequential_chain() -> CompositionChain:
    """Create a more complex 5-model sequential chain"""
    
    print("⛓️  Creating Complex Sequential Processing Chain")
    print("   Multi-step processing with 5 sequential models")
    print("   Mode: Sequential On-Chain Execution")
    
    chain = CompositionChain(
        chain_id="complex_sequential_chain",
        description="Complex 5-step sequential processing chain"
    )
    
    # Add multiple models for complex processing
    chain.add_model(ComposableFeatureExtractor())
    chain.add_model(ComposableWeatherProcessor())
    chain.add_model(ComposableSequenceModel())
    chain.add_model(ComposableClassifier())
    chain.add_model(ComposableDecisionMaker())
    
    if chain.finalize():
        print("✅ Complex sequential chain created and validated")
        return chain
    else:
        print("❌ Complex sequential chain validation failed")
        return None

def run_sequential_composition(chain: CompositionChain, output_dir: str = None):
    """Run the complete sequential composition pipeline"""
    
    print(f"\\n🚀 Starting Sequential Proof Composition")
    print("=" * 60)
    
    # Initialize the sequential pipeline
    pipeline = SequentialProofCompositionPipeline(
        chain=chain,
        output_dir=output_dir
    )
    
    try:
        # Run the complete sequential composition workflow
        success = pipeline.run_complete_sequential_composition()
        
        if success:
            print("\\n🎉 Sequential Composition Pipeline Completed Successfully!")
            print("=" * 60)
            print(f"📁 Output directory: {pipeline.output_dir}")
            print("\\n📋 Generated Files:")
            
            import os
            if os.path.exists(pipeline.output_dir):
                for file in sorted(os.listdir(pipeline.output_dir)):
                    if file.endswith(('.sol', '.js', '.json', '.md')):
                        print(f"   📄 {file}")
            
            print("\\n🚀 Next Steps:")
            print("   1. Copy contracts to your Hardhat contracts directory")
            print("   2. Run: npx hardhat compile")
            print("   3. Run: npx hardhat run deploy_sequential_chain.js --network localhost")
            print("   4. Run: npx hardhat run test_sequential_chain.js --network localhost")
            
        else:
            print("\\n❌ Sequential Composition Pipeline Failed")
            
    except Exception as e:
        print(f"\\n💥 Pipeline error: {e}")
        return False
    finally:
        # Cleanup temporary files
        pipeline.cleanup()
    
    return success

def main():
    """Main demo function"""
    
    parser = argparse.ArgumentParser(description="Sequential Proof Composition Pipeline Demo")
    parser.add_argument(
        "--chain-type", 
        choices=["simple", "weather", "complex"],
        default="simple",
        help="Type of sequential chain to create"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for generated artifacts"
    )
    
    args = parser.parse_args()
    
    print("⛓️  Sequential Proof Composition Pipeline Demo")
    print("=" * 60)
    print("Mode: Sequential On-Chain Execution")
    print("Key Features:")
    print("  • Each step feeds output to next step's input") 
    print("  • True sequential verification on-chain")
    print("  • Intermediate values visible and debuggable")
    print("  • Atomic execution (all steps succeed or fail)")
    print("=" * 60)
    
    # Create the specified chain type
    if args.chain_type == "simple":
        chain = create_simple_sequential_chain()
    elif args.chain_type == "weather":
        chain = create_weather_sequential_chain()
    elif args.chain_type == "complex":
        chain = create_complex_sequential_chain()
    else:
        print(f"❌ Unknown chain type: {args.chain_type}")
        return
    
    if not chain:
        print("❌ Failed to create chain")
        return
    
    # Run the sequential composition
    success = run_sequential_composition(chain, args.output_dir)
    
    if success:
        print("\\n✅ Demo completed successfully!")
    else:
        print("\\n❌ Demo failed!")

if __name__ == "__main__":
    main()