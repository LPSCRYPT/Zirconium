#!/usr/bin/env python3
"""Debug verification step by step"""

from web3 import Web3
import json

# Connect to localhost
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
verifier_address = "0xccA9728291bC98ff4F97EF57Be3466227b0eb06C"

# Load ABI and calldata
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/weather_verifier_standard_abi.json', 'r') as f:
    abi = json.load(f)

with open('/Users/bot/code/zirconium/ezkl_weather_workspace/standard_calldata.bytes', 'rb') as f:
    calldata = f.read()

contract = web3.eth.contract(address=verifier_address, abi=abi)

print("=== TEST 1: Empty instances array ===")
try:
    result = contract.functions.verifyProof(calldata, []).call()
    print(f"✅ Empty instances: {result}")
except Exception as e:
    print(f"❌ Empty instances failed: {str(e)[:100]}...")

print("\n=== TEST 2: Single zero instance ===")
try:
    result = contract.functions.verifyProof(calldata, [0]).call()
    print(f"✅ Single zero: {result}")
except Exception as e:
    print(f"❌ Single zero failed: {str(e)[:100]}...")

print("\n=== TEST 3: Using witness inputs instead of outputs ===")
# Extract inputs from witness (10 values)
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/witness.json', 'r') as f:
    witness = json.load(f)

# Try inputs as public instances
input_instances = []
for input_hex in witness['inputs'][0]:
    input_instances.append(int(input_hex, 16))

print(f"Input instances: {len(input_instances)} values")
try:
    result = contract.functions.verifyProof(calldata, input_instances).call()
    print(f"✅ Using inputs: {result}")
except Exception as e:
    print(f"❌ Using inputs failed: {str(e)[:100]}...")

print("\n=== TEST 4: Using witness outputs (original approach) ===")
# Extract outputs from witness (4 values) 
output_instances = []
for output_hex in witness['outputs'][0]:
    output_instances.append(int(output_hex, 16))

print(f"Output instances: {len(output_instances)} values")
try:
    result = contract.functions.verifyProof(calldata, output_instances).call()
    print(f"✅ Using outputs: {result}")
except Exception as e:
    print(f"❌ Using outputs failed: {str(e)[:100]}...")

print("\n=== TEST 5: Using all instances (inputs + outputs) ===")
all_instances = input_instances + output_instances
print(f"All instances: {len(all_instances)} values")
try:
    result = contract.functions.verifyProof(calldata, all_instances).call()
    print(f"✅ Using all: {result}")
except Exception as e:
    print(f"❌ Using all failed: {str(e)[:100]}...")