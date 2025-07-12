#!/usr/bin/env python3
"""Extract public instances from xlstm_simple witness"""

import json

# Manually copied witness data for xlstm_simple (first few lines to check structure)
witness_inputs = [
    "ffffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
    "fcffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430", 
    "0500000000000000000000000000000000000000000000000000000000000000",
    "0000000000000000000000000000000000000000000000000000000000000000",
    "fcffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "feffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
    "0200000000000000000000000000000000000000000000000000000000000000",
    "0200000000000000000000000000000000000000000000000000000000000000",
    "fdffffef93f5e1439170b97948e833285d588181b64550b829a031e1724e6430"
]

witness_outputs = [
    "0200000000000000000000000000000000000000000000000000000000000000",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "0100000000000000000000000000000000000000000000000000000000000000",
    "0200000000000000000000000000000000000000000000000000000000000000"
    # ... (truncated for brevity, real witness has 69 outputs)
]

# Convert hex to integers for the first 5 inputs and 5 outputs as a test
print("=== xlstm_simple PUBLIC INSTANCES TEST ===")

input_instances = []
for i, hex_val in enumerate(witness_inputs):
    int_val = int(hex_val, 16)
    input_instances.append(int_val)
    print(f"Input[{i}]: {hex_val} -> {int_val}")

print()

output_instances = []
for i, hex_val in enumerate(witness_outputs):
    int_val = int(hex_val, 16) 
    output_instances.append(int_val)
    print(f"Output[{i}]: {hex_val} -> {int_val}")

# For xlstm_simple model, ALL inputs and outputs are public instances
test_public_instances = input_instances + output_instances
print(f"\nTest public instances count: {len(test_public_instances)} (first 10 inputs + first 5 outputs)")
print(f"Expected total for full model: 79 (10 inputs + 69 outputs)")

# Save minimal test set
with open('test_public_instances.json', 'w') as f:
    json.dump(test_public_instances, f)

print("✅ Test instances saved to test_public_instances.json")