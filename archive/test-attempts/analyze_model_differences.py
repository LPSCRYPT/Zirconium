#!/usr/bin/env python3
"""Analyze differences between xlstm_simple and weather models"""

import json

print("=== MODEL COMPARISON ANALYSIS ===")

# Load xlstm_simple settings
with open('/Users/bot/code/zirconium/ezkl_workspace/xlstm_simple/settings.json', 'r') as f:
    simple_settings = json.load(f)

# Load weather model settings  
with open('/Users/bot/code/zirconium/ezkl_weather_workspace/settings.json', 'r') as f:
    weather_settings = json.load(f)

print("XLSTM_SIMPLE MODEL:")
print(f"  - Input visibility: {simple_settings['run_args']['input_visibility']}")
print(f"  - Output visibility: {simple_settings['run_args']['output_visibility']}")
print(f"  - Logrows: {simple_settings['run_args']['logrows']}")
print(f"  - Model instance shapes: {simple_settings['model_instance_shapes']}")
print(f"  - Total rows: {simple_settings['num_rows']}")

print("\nWEATHER MODEL:")
print(f"  - Input visibility: {weather_settings['run_args']['input_visibility']}")
print(f"  - Output visibility: {weather_settings['run_args']['output_visibility']}")
print(f"  - Logrows: {weather_settings['run_args']['logrows']}")
print(f"  - Model instance shapes: {weather_settings['model_instance_shapes']}")
print(f"  - Total rows: {weather_settings['num_rows']}")

print("\nKEY DIFFERENCES:")
print("1. INPUT VISIBILITY:")
print(f"   Simple: {simple_settings['run_args']['input_visibility']} (inputs are public instances)")
print(f"   Weather: {weather_settings['run_args']['input_visibility']} (inputs are private)")

print("2. CIRCUIT COMPLEXITY:")
print(f"   Simple: {simple_settings['num_rows']} rows, logrows={simple_settings['run_args']['logrows']}")
print(f"   Weather: {weather_settings['num_rows']} rows, logrows={weather_settings['run_args']['logrows']}")

print("3. PUBLIC INSTANCES COUNT:")
simple_shapes = simple_settings['model_instance_shapes']
weather_shapes = weather_settings['model_instance_shapes']
print(f"   Simple: {simple_shapes} -> inputs + outputs = {sum(shape[1] for shape in simple_shapes)} public instances")
print(f"   Weather: {weather_shapes} -> outputs only = {weather_shapes[0][1]} public instances")

print("\nHYPOTHESIS:")
print("The weather model fails blockchain verification because:")
print("1. It has very few public instances (4) compared to simple model (79)")
print("2. It uses private inputs, which may create different proof structure")  
print("3. It's much more complex (25156 rows vs 4854 rows)")
print("4. The verifier contract may expect the public input pattern from simple model")

print("\nNEXT STEPS:")
print("1. Generate proper verifier contract for xlstm_simple model")
print("2. Test xlstm_simple blockchain verification") 
print("3. If it works, adapt our weather model to match simple model's public visibility pattern")
print("4. Consider simplifying weather model architecture")