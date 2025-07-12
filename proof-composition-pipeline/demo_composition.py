#!/usr/bin/env python3
"""
Proof Composition Pipeline Demo

Demonstrates composing multiple models into verified chains.
"""

import sys
import os
import argparse

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'interfaces'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'core', 'pipeline'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'chains'))

from composable_model import CompositionChain
from composition_pipeline import ProofCompositionPipeline
from simple_chain_models import (
    ComposableFeatureExtractor,
    ComposableClassifier, 
    ComposableDecisionMaker,
    ComposableWeatherProcessor,
    ComposableSequenceModel
)

def create_simple_chain() -> CompositionChain:
    """Create a simple 3-model chain: Feature Extraction -> Classification -> Decision"""
    
    print("🔗 Creating Simple Processing Chain")
    print("   Feature Extraction -> Classification -> Decision Making")
    
    chain = CompositionChain(
        chain_id="simple_processing_chain",
        description="Feature extraction followed by classification and decision making"
    )
    
    # Add models in sequence
    chain.add_model(ComposableFeatureExtractor())
    chain.add_model(ComposableClassifier())
    chain.add_model(ComposableDecisionMaker())
    
    # Finalize and validate
    if chain.finalize():
        print("✅ Simple chain created and validated")
        return chain
    else:
        print("❌ Simple chain validation failed")
        return None

def create_weather_chain() -> CompositionChain:
    """Create a weather-focused chain: Weather Processing -> Sequence Processing -> Decision"""
    
    print("🌤️ Creating Weather Processing Chain")
    print("   Weather Processing -> Sequence Processing -> Decision Making")
    
    chain = CompositionChain(
        chain_id="weather_analysis_chain", 
        description="Weather data processing with temporal analysis and decision making"
    )
    
    # Add models in sequence
    chain.add_model(ComposableWeatherProcessor())
    chain.add_model(ComposableSequenceModel())
    chain.add_model(ComposableDecisionMaker())
    
    # Finalize and validate
    if chain.finalize():
        print("✅ Weather chain created and validated")
        return chain
    else:
        print("❌ Weather chain validation failed")
        return None

def create_complex_chain() -> CompositionChain:
    """Create a more complex 4-model chain"""
    
    print("🔄 Creating Complex Processing Chain")
    print("   Feature Extraction -> Weather Processing -> Sequence Processing -> Classification")
    
    chain = CompositionChain(
        chain_id="complex_processing_chain",
        description="Complex multi-stage processing with feature extraction, weather processing, sequence analysis, and classification"
    )
    
    # Add models in sequence
    chain.add_model(ComposableFeatureExtractor())
    chain.add_model(ComposableWeatherProcessor()) 
    chain.add_model(ComposableSequenceModel())
    chain.add_model(ComposableClassifier())
    
    # Finalize and validate
    if chain.finalize():
        print("✅ Complex chain created and validated")
        return chain
    else:
        print("❌ Complex chain validation failed")
        return None

def run_composition_demo(chain_type: str = "simple", generate_blockchain: bool = False):
    """Run the proof composition demo"""
    
    print("🚀 Proof Composition Pipeline Demo")
    print("=" * 60)
    
    # Create the specified chain
    if chain_type == "simple":
        chain = create_simple_chain()
    elif chain_type == "weather":
        chain = create_weather_chain()
    elif chain_type == "complex":
        chain = create_complex_chain()
    else:
        print(f"❌ Unknown chain type: {chain_type}")
        return False
    
    if not chain:
        return False
    
    print(f"\n📊 Chain Information:")
    print(f"   ID: {chain.chain_id}")
    print(f"   Length: {len(chain)} models")
    print(f"   Description: {chain.description}")
    
    # Print chain details
    print(f"\n🔗 Chain Composition:")
    for i, position in enumerate(chain):
        model_config = position.model.get_config()
        print(f"   {i+1}. {model_config['name']} ({model_config['architecture']})")
        print(f"      Input: {position.model.get_input_shape()}")
        print(f"      Output: {position.model.get_output_shape()}")
    
    # Ask for confirmation
    if generate_blockchain:
        print(f"\n⚠️  This will generate full blockchain verification artifacts.")
        print(f"   This may take several minutes...")
    
    # Initialize and run the composition pipeline
    pipeline = ProofCompositionPipeline(
        chain=chain,
        output_dir=f"outputs/pcp_{chain.chain_id}"
    )
    
    try:
        success = pipeline.run_complete_composition()
        
        if success:
            print(f"\n🎉 Proof composition completed successfully!")
            print(f"📁 Results available in: {pipeline.output_dir}")
            
            if generate_blockchain:
                print(f"\n🔗 Blockchain Integration:")
                print(f"   1. Deploy contracts: npx hardhat run {pipeline.output_dir}/deploy_chain.js --network localhost")
                print(f"   2. Test verification: npx hardhat run {pipeline.output_dir}/test_chain.js --network localhost")
        else:
            print(f"\n❌ Proof composition failed")
            
        return success
        
    except Exception as e:
        print(f"\n💥 Error during composition: {e}")
        return False
    
    finally:
        # Cleanup temporary files
        pipeline.cleanup()

def list_available_chains():
    """List all available chain types"""
    
    print("🔧 Available Chain Types:")
    print("-" * 40)
    print("   simple    - Feature Extraction -> Classification -> Decision (3 models)")
    print("   weather   - Weather Processing -> Sequence Processing -> Decision (3 models)")
    print("   complex   - Feature -> Weather -> Sequence -> Classification (4 models)")

def main():
    parser = argparse.ArgumentParser(description="Proof Composition Pipeline Demo")
    parser.add_argument("--chain", "-c", 
                       choices=["simple", "weather", "complex"],
                       default="simple",
                       help="Type of chain to create")
    parser.add_argument("--blockchain", "-b",
                       action="store_true",
                       help="Generate blockchain verification artifacts")
    parser.add_argument("--list", "-l",
                       action="store_true", 
                       help="List available chain types")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_chains()
        return
    
    success = run_composition_demo(
        chain_type=args.chain,
        generate_blockchain=args.blockchain
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()