#!/usr/bin/env python3
"""Test blockchain verification with xlstm_simple model data"""

from web3 import Web3
import json

print("=== Testing xlstm_simple Model Blockchain Verification ===")

# Connect to localhost blockchain
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

if not web3.is_connected():
    print("❌ Cannot connect to localhost blockchain")
    print("Please make sure: npx hardhat node is running")
    exit(1)

print("✅ Connected to blockchain")

# Use existing weather verifier for initial test (we'll adapt later)
verifier_address = "0xccA9728291bC98ff4F97EF57Be3466227b0eb06C"

# Load the ABI (use weather verifier ABI for now)
try:
    with open('/Users/bot/code/zirconium/ezkl_weather_workspace/weather_verifier_standard_abi.json', 'r') as f:
        abi = json.load(f)
    print("✅ Loaded verifier ABI")
except Exception as e:
    print(f"❌ Failed to load ABI: {e}")
    exit(1)

# Create contract instance
contract = web3.eth.contract(address=verifier_address, abi=abi)

# Test with xlstm_simple style data (simplified)
# For the simple model: both inputs and outputs are public
# Converting some sample hex values from the witness

test_instances = [
    # Sample inputs converted to integers (simplified)
    int("ffffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430", 16),
    int("fcffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430", 16), 
    5,  # Simple small values
    0,
    # Sample outputs (also simplified)
    2,
    1,
    1,
    1
]

print(f"Test instances: {len(test_instances)} values")
for i, val in enumerate(test_instances):
    print(f"  [{i}]: {val}")

# Load weather model calldata for structure comparison
try:
    with open('/Users/bot/code/zirconium/ezkl_weather_workspace/standard_calldata.bytes', 'rb') as f:
        weather_calldata = f.read()
    print(f"✅ Loaded weather calldata ({len(weather_calldata)} bytes)")
except Exception as e:
    print(f"❌ Failed to load weather calldata: {e}")
    exit(1)

print("\n=== TEST 1: Empty instances with weather calldata ===")
try:
    result = contract.functions.verifyProof(weather_calldata, []).call()
    print(f"✅ Empty instances: {result}")
except Exception as e:
    print(f"❌ Empty instances failed: {str(e)[:100]}...")

print("\n=== TEST 2: Test instances with weather calldata ===")
try:
    result = contract.functions.verifyProof(weather_calldata, test_instances).call()
    print(f"✅ Test instances: {result}")
except Exception as e:
    print(f"❌ Test instances failed: {str(e)[:100]}...")

print("\n=== FINDINGS ===")
print("- The xlstm_simple model has both inputs and outputs as public instances")
print("- Our weather model only has outputs as public instances") 
print("- The simple model has 79 total public instances vs our 4")
print("- Next step: Generate proper verifier contract for xlstm_simple model")
print("- This test shows the approach - need proper calldata for simple model")