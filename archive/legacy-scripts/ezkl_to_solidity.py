#!/usr/bin/env python3
"""
Convert EZKL proof format to Solidity verifyProof function parameters
"""
import json
import ezkl

def convert_ezkl_proof_to_solidity(proof_path):
    """
    Convert EZKL proof to Solidity verifyProof function parameters
    
    Args:
        proof_path: Path to the EZKL proof.json file
        
    Returns:
        tuple: (proof_bytes, instances_array)
            - proof_bytes: bytes object for Solidity bytes calldata
            - instances_array: list of integers for Solidity uint256[] calldata
    """
    
    # Load the original proof structure
    with open(proof_path, 'r') as f:
        proof_data = json.load(f)
    
    print("🔍 EZKL Proof Analysis")
    print("=" * 25)
    print(f"Proof keys: {list(proof_data.keys())}")
    print(f"Instances: {proof_data['instances']}")
    print(f"Number of public inputs: {len(proof_data['instances'][0])}")
    
    # Try using raw proof bytes directly
    raw_proof = proof_data['proof']
    proof_bytes_raw = bytes(raw_proof)
    
    # Also try EZKL's encode_evm_calldata function to get proper bytes format
    calldata_bytes = ezkl.encode_evm_calldata(proof_path, 'temp_calldata.json')
    
    # Extract public inputs (instances) from the proof
    instances_hex = proof_data['instances'][0]  # First (and only) instance array
    instances_uint256 = []
    
    for hex_str in instances_hex:
        # Convert hex string to integer
        if hex_str.startswith('0x'):
            instances_uint256.append(int(hex_str, 16))
        else:
            instances_uint256.append(int(hex_str, 16))
    
    print(f"\n🔐 Solidity Format Conversion")
    print("=" * 30)
    print(f"Raw proof bytes length: {len(proof_bytes_raw)}")
    print(f"Raw proof bytes (first 32): {proof_bytes_raw[:32].hex()}")
    print(f"Calldata bytes length: {len(calldata_bytes)}")
    print(f"Calldata bytes (first 32): {calldata_bytes[:32].hex()}")
    print(f"Instances array: {instances_uint256}")
    print(f"Instances count: {len(instances_uint256)}")
    
    # Try both formats - return encoded calldata for reusable verifier
    return calldata_bytes, instances_uint256

def test_conversion():
    """Test the conversion function"""
    proof_path = "/Users/bot/code/zirconium/ezkl_weather_workspace/proof.json"
    
    try:
        proof_bytes, instances = convert_ezkl_proof_to_solidity(proof_path)
        
        print(f"\n✅ Conversion successful!")
        print(f"Proof bytes type: {type(proof_bytes)}")
        print(f"Instances type: {type(instances)}")
        
        # Verify the format is correct
        assert isinstance(proof_bytes, bytes), "Proof should be bytes"
        assert isinstance(instances, list), "Instances should be list"
        assert all(isinstance(x, int) for x in instances), "All instances should be integers"
        
        print(f"✅ Format validation passed!")
        
        return proof_bytes, instances
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return None, None

if __name__ == "__main__":
    test_conversion()