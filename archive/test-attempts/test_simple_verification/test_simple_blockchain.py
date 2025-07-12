#!/usr/bin/env python3
"""Test xlstm_simple model blockchain verification"""

import os
import shutil
from web3 import Web3
import json

# Set working directory
os.chdir('/Users/bot/code/zirconium/test_simple_verification')

print("=== STEP 1: Copy files from xlstm_simple ===")
source_dir = '/Users/bot/code/zirconium/ezkl_workspace/xlstm_simple'

# Copy essential files
files_to_copy = ['vk.key', 'proof.json', 'witness.json', 'calldata.txt']
for file in files_to_copy:
    src = f"{source_dir}/{file}"
    dst = file
    try:
        shutil.copy2(src, dst)
        print(f"✅ Copied {file}")
    except Exception as e:
        print(f"❌ Failed to copy {file}: {e}")

print("\n=== STEP 2: Generate EVM verifier contract ===")
import subprocess

try:
    # Generate verifier contract using EZKL
    result = subprocess.run([
        'ezkl', 'create-evm-verifier',
        '--vk-path', 'vk.key',
        '--srs-path', '/Users/bot/.ezkl/srs/kzg12.srs',  # logrows=12 for simple model
        '--settings-path', 'settings.json',
        '--sol-code-path', 'simple_verifier.sol',
        '--abi-path', 'simple_verifier_abi.json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Verifier contract generated")
    else:
        print(f"❌ Verifier generation failed: {result.stderr}")
        
except Exception as e:
    print(f"❌ Error generating verifier: {e}")

print("\n=== STEP 3: Generate calldata ===")
try:
    result = subprocess.run([
        'ezkl', 'encode-evm-calldata',
        '--proof-path', 'proof.json',
        '--calldata-path', 'simple_calldata.bytes'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Calldata generated")
    else:
        print(f"❌ Calldata generation failed: {result.stderr}")
        
except Exception as e:
    print(f"❌ Error generating calldata: {e}")

print("\n=== STEP 4: Prepare public instances ===")
# Load witness and extract ALL public instances (inputs + outputs)
with open('witness.json', 'r') as f:
    witness = json.load(f)

# For xlstm_simple: both inputs and outputs are public
public_instances = []

# Add inputs (10 values)
for input_hex in witness['inputs'][0]:
    public_instances.append(int(input_hex, 16))

# Add outputs (69 values) 
for output_hex in witness['outputs'][0]:
    public_instances.append(int(output_hex, 16))

print(f"Total public instances: {len(public_instances)} (10 inputs + 69 outputs)")

# Save for blockchain test
with open('public_instances.json', 'w') as f:
    json.dump(public_instances, f)

print("✅ Setup complete - ready for blockchain deployment test")