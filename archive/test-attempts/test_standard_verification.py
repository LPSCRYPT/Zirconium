#!/usr/bin/env python3
"""Test standard verifier with standard proof"""

from web3 import Web3
import json

# Connect to localhost
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))

# Contract address from deployment
verifier_address = "0xccA9728291bC98ff4F97EF57Be3466227b0eb06C"

# Load the ABI
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/weather_verifier_standard_abi.json', 'r') as f:
    abi = json.load(f)

# Load calldata
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/standard_calldata.bytes', 'rb') as f:
    calldata = f.read()

print(f"Calldata size: {len(calldata)} bytes")
print(f"First 64 bytes (hex): {calldata[:64].hex()}")

# Create contract instance
contract = web3.eth.contract(address=verifier_address, abi=abi)

# Test the verification - but first let's see what functions are available
print("\nAvailable contract functions:")
for func in contract.all_functions():
    print(f"  {func.fn_name}")

# Load witness to get public instances
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/witness.json', 'r') as f:
    witness = json.load(f)

# Extract public instances (outputs) - convert hex strings to integers
public_instances = []
for output_hex in witness['outputs'][0]:
    # Convert from hex string to int
    instance_int = int(output_hex, 16)
    public_instances.append(instance_int)

print(f"Public instances: {len(public_instances)} values")
for i, instance in enumerate(public_instances):
    print(f"  [{i}]: {instance}")

# Test verification with proper parameters
try:
    result = contract.functions.verifyProof(calldata, public_instances).call()
    print(f"\nVerification result: {result}")
except Exception as e:
    print(f"\nVerification failed: {e}")
    
    # Try with transaction instead of call to see more details
    try:
        print("Trying as transaction...")
        tx_hash = contract.functions.verifyProof(calldata, public_instances).transact()
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Transaction successful: {receipt.status}")
    except Exception as tx_e:
        print(f"Transaction also failed: {tx_e}")