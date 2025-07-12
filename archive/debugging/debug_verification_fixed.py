#!/usr/bin/env python3
"""Debug verification with field element correction"""

from web3 import Web3
import json

# BN254 field modulus  
BN254_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617

# Connect to localhost
web3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
verifier_address = "0xccA9728291bC98ff4F97EF57Be3466227b0eb06C"

# Load ABI and calldata
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/weather_verifier_standard_abi.json', 'r') as f:
    abi = json.load(f)

with open('/Users/bot/code/zirconium/ezkl_weather_workspace/standard_calldata.bytes', 'rb') as f:
    calldata = f.read()

contract = web3.eth.contract(address=verifier_address, abi=abi)

# Load witness and extract outputs
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/witness.json', 'r') as f:
    witness = json.load(f)

# Extract outputs and apply modular reduction
output_instances_raw = []
for output_hex in witness['outputs'][0]:
    output_instances_raw.append(int(output_hex, 16))

output_instances_reduced = []
for val in output_instances_raw:
    reduced = val % BN254_MODULUS
    output_instances_reduced.append(reduced)

print("=== RAW vs REDUCED VALUES ===")
for i, (raw, reduced) in enumerate(zip(output_instances_raw, output_instances_reduced)):
    print(f"[{i}] Raw: {raw}")
    print(f"[{i}] Reduced: {reduced}")
    print(f"[{i}] Same?: {raw == reduced}")
    print()

print("=== TEST: Using modular reduced instances ===")
try:
    result = contract.functions.verifyProof(calldata, output_instances_reduced).call()
    print(f"✅ Reduced instances verification: {result}")
except Exception as e:
    print(f"❌ Reduced instances failed: {str(e)[:200]}...")

print("\n=== TEST: Using original instances for comparison ===")
try:
    result = contract.functions.verifyProof(calldata, output_instances_raw).call()
    print(f"✅ Raw instances verification: {result}")
except Exception as e:
    print(f"❌ Raw instances failed: {str(e)[:200]}...")