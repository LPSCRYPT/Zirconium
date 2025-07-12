#!/usr/bin/env python3
"""Test EZKL Python verification before blockchain"""

import ezkl
import os

os.chdir('/Users/bot/code/zirconium/ezkl_weather_workspace')

print("=== Testing EZKL Python Verification ===")

try:
    # Test the proof verification with EZKL Python directly
    result = ezkl.verify(
        proof_path="proof.json",
        settings_path="settings.json", 
        vk_path="vk.key"
    )
    print(f"✅ EZKL Python verification: {result}")
except Exception as e:
    print(f"❌ EZKL Python verification failed: {e}")

print("\n=== Testing EVM Verification ===")
try:
    # Test if EZKL has built-in EVM verification
    result = ezkl.verify_evm(
        proof_path="proof.json",
        settings_path="settings.json",
        vk_path="vk.key"
    )
    print(f"✅ EZKL EVM verification: {result}")
except Exception as e:
    print(f"❌ EZKL EVM verification failed: {e}")

print("\n=== Checking Proof Structure ===")
import json
with open('proof.json', 'r') as f:
    proof = json.load(f)
    
print(f"Proof keys: {list(proof.keys())}")
if 'instances' in proof:
    print(f"Proof instances: {proof['instances']}")
if 'transcript_type' in proof:
    print(f"Transcript type: {proof['transcript_type']}")
if 'protocol' in proof:
    print(f"Protocol keys: {list(proof['protocol'].keys()) if isinstance(proof['protocol'], dict) else 'Not dict'}")