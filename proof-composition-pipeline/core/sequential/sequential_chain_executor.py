#!/usr/bin/env python3
"""
Sequential Chain Executor

Implements true sequential on-chain verification where:
1. Model A verifier runs with initial input, produces output
2. Model B verifier runs with A's output, produces its output  
3. Model C verifier runs with B's output, produces final output
4. Each step is verified independently on-chain in sequence
"""

import os
import sys
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Add interfaces to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interfaces'))
from composable_model import ComposableModelInterface, CompositionChain

@dataclass
class SequentialStep:
    """Represents a single step in sequential execution"""
    step_id: int
    model_name: str
    verifier_contract_address: str
    input_data: List[List[float]]
    expected_output: List[List[float]]
    proof_data: bytes
    calldata: bytes

class SequentialChainExecutor:
    """Generates contracts for true sequential on-chain verification"""
    
    def __init__(self, chain: CompositionChain):
        self.chain = chain
        self.steps: List[SequentialStep] = []
        
    def generate_sequential_contracts(self, output_dir: str) -> Dict[str, str]:
        """Generate contracts for sequential execution"""
        
        contracts = {}
        
        # Generate the main sequential executor contract
        executor_contract = self._generate_sequential_executor_contract()
        contracts["SequentialChainExecutor.sol"] = executor_contract
        
        # Generate enhanced individual verifiers that return outputs
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            
            enhanced_verifier = self._generate_enhanced_verifier_contract(i, model_name, model)
            contracts[f"Enhanced{model_name.title()}Verifier.sol"] = enhanced_verifier
            
        return contracts
        
    def _generate_sequential_executor_contract(self) -> str:
        """Generate the main sequential executor contract"""
        
        step_contracts = []
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            step_contracts.append({
                "step": i,
                "name": model_name,
                "contract_name": f"Enhanced{model_name.title()}Verifier"
            })
        
        contract = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Sequential Chain Executor
 * 
 * Executes a {len(self.chain)}-step proof composition chain sequentially:
 * {' -> '.join([pos.model.get_config()['name'] for pos in self.chain])}
 * 
 * Each step verifies a proof AND produces output for the next step.
 */

interface IEnhancedVerifier {{
    function verifyAndExecute(
        bytes calldata proofData,
        uint256[] calldata publicInputs
    ) external returns (bool success, uint256[] memory outputs);
}}

contract SequentialChainExecutor {{
    
    struct ChainStep {{
        address verifierContract;
        string stepName;
        uint256 expectedInputSize;
        uint256 expectedOutputSize;
    }}
    
    struct ExecutionResult {{
        bool success;
        uint256 totalSteps;
        uint256[][] intermediateOutputs;
        uint256[] finalOutput;
        uint256 totalGasUsed;
    }}
    
    ChainStep[] public chainSteps;
    
    event StepExecuted(
        uint256 indexed stepIndex, 
        string stepName, 
        bool success,
        uint256[] output,
        uint256 gasUsed
    );
    
    event ChainCompleted(
        bool success,
        uint256 totalSteps,
        uint256[] finalOutput,
        uint256 totalGasUsed
    );
    
    constructor() {{
        // Initialize chain steps - addresses set after individual deployments
    }}
    
    function setChainSteps(
        address[] memory verifierAddresses,
        string[] memory stepNames,
        uint256[] memory inputSizes,
        uint256[] memory outputSizes
    ) external {{
        require(verifierAddresses.length == stepNames.length, "Mismatched arrays");
        require(verifierAddresses.length == inputSizes.length, "Mismatched arrays");
        require(verifierAddresses.length == outputSizes.length, "Mismatched arrays");
        
        delete chainSteps;
        
        for (uint256 i = 0; i < verifierAddresses.length; i++) {{
            chainSteps.push(ChainStep({{
                verifierContract: verifierAddresses[i],
                stepName: stepNames[i],
                expectedInputSize: inputSizes[i],
                expectedOutputSize: outputSizes[i]
            }}));
        }}
    }}
    
    function executeSequentialChain(
        uint256[] memory initialInput,
        bytes[] memory proofData
    ) external returns (ExecutionResult memory) {{
        
        require(chainSteps.length > 0, "Chain not configured");
        require(proofData.length == chainSteps.length, "Proof count mismatch");
        
        uint256 startGas = gasleft();
        uint256[][] memory intermediateOutputs = new uint256[][](chainSteps.length);
        uint256[] memory currentInput = initialInput;
        bool allSuccessful = true;
        
        // Execute each step sequentially
        for (uint256 i = 0; i < chainSteps.length; i++) {{
            ChainStep memory step = chainSteps[i];
            
            // Verify input size matches expectation
            require(currentInput.length == step.expectedInputSize, 
                string(abi.encodePacked("Step ", toString(i), " input size mismatch")));
            
            uint256 stepStartGas = gasleft();
            
            // Call the enhanced verifier
            (bool stepSuccess, uint256[] memory stepOutput) = IEnhancedVerifier(step.verifierContract)
                .verifyAndExecute(proofData[i], currentInput);
            
            uint256 stepGasUsed = stepStartGas - gasleft();
            
            if (!stepSuccess) {{
                allSuccessful = false;
                emit StepExecuted(i, step.stepName, false, new uint256[](0), stepGasUsed);
                break;
            }}
            
            // Verify output size matches expectation
            require(stepOutput.length == step.expectedOutputSize,
                string(abi.encodePacked("Step ", toString(i), " output size mismatch")));
            
            // Store intermediate output
            intermediateOutputs[i] = stepOutput;
            
            // Output becomes input for next step
            currentInput = stepOutput;
            
            emit StepExecuted(i, step.stepName, true, stepOutput, stepGasUsed);
        }}
        
        uint256 totalGasUsed = startGas - gasleft();
        
        ExecutionResult memory result = ExecutionResult({{
            success: allSuccessful,
            totalSteps: chainSteps.length,
            intermediateOutputs: intermediateOutputs,
            finalOutput: allSuccessful ? currentInput : new uint256[](0),
            totalGasUsed: totalGasUsed
        }});
        
        emit ChainCompleted(allSuccessful, chainSteps.length, result.finalOutput, totalGasUsed);
        
        return result;
    }}
    
    function getChainInfo() external view returns (
        uint256 stepCount,
        string[] memory stepNames,
        address[] memory verifierAddresses
    ) {{
        stepCount = chainSteps.length;
        stepNames = new string[](stepCount);
        verifierAddresses = new address[](stepCount);
        
        for (uint256 i = 0; i < stepCount; i++) {{
            stepNames[i] = chainSteps[i].stepName;
            verifierAddresses[i] = chainSteps[i].verifierContract;
        }}
    }}
    
    function getStepInfo(uint256 stepIndex) external view returns (
        address verifierContract,
        string memory stepName,
        uint256 expectedInputSize,
        uint256 expectedOutputSize
    ) {{
        require(stepIndex < chainSteps.length, "Invalid step index");
        ChainStep memory step = chainSteps[stepIndex];
        return (step.verifierContract, step.stepName, step.expectedInputSize, step.expectedOutputSize);
    }}
    
    // Helper function to convert uint to string
    function toString(uint256 value) internal pure returns (string memory) {{
        if (value == 0) return "0";
        
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {{
            digits++;
            temp /= 10;
        }}
        
        bytes memory buffer = new bytes(digits);
        while (value != 0) {{
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }}
        
        return string(buffer);
    }}
}}"""
        
        return contract
        
    def _generate_enhanced_verifier_contract(self, step_index: int, model_name: str, 
                                          model: ComposableModelInterface) -> str:
        """Generate enhanced verifier that returns outputs"""
        
        input_shape = model.get_input_shape()
        output_shape = model.get_output_shape()
        
        contract = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Enhanced {model_name.title()} Verifier
 * 
 * Step {step_index} in the sequential chain.
 * Input Shape: {input_shape}
 * Output Shape: {output_shape}
 * 
 * This contract both verifies the proof AND returns the computation output
 * for use by the next step in the chain.
 */

import "./step{step_index:02d}_{model_name}Verifier.sol";

contract Enhanced{model_name.title()}Verifier {{
    
    // Reference to the underlying EZKL verifier
    address public immutable underlyingVerifier;
    
    // Expected input/output sizes
    uint256 public constant INPUT_SIZE = {input_shape[0] if input_shape else 0};
    uint256 public constant OUTPUT_SIZE = {output_shape[0] if output_shape else 0};
    
    event ProofVerified(bool success, uint256[] output);
    event VerificationFailed(string reason);
    
    constructor(address _underlyingVerifier) {{
        underlyingVerifier = _underlyingVerifier;
    }}
    
    /**
     * Verify proof and return computation output
     * 
     * @param proofData The EZKL proof data
     * @param publicInputs The public inputs (includes both input and expected output)
     * @return success Whether verification passed
     * @return outputs The computed outputs (extracted from public inputs)
     */
    function verifyAndExecute(
        bytes calldata proofData,
        uint256[] calldata publicInputs
    ) external returns (bool success, uint256[] memory outputs) {{
        
        // Validate input size
        if (publicInputs.length < INPUT_SIZE) {{
            emit VerificationFailed("Insufficient public inputs");
            return (false, new uint256[](0));
        }}
        
        // Extract inputs and expected outputs from publicInputs
        // Format: [input_0, input_1, ..., input_n, output_0, output_1, ..., output_m]
        uint256[] memory inputs = new uint256[](INPUT_SIZE);
        uint256[] memory expectedOutputs = new uint256[](OUTPUT_SIZE);
        
        for (uint256 i = 0; i < INPUT_SIZE; i++) {{
            inputs[i] = publicInputs[i];
        }}
        
        for (uint256 i = 0; i < OUTPUT_SIZE; i++) {{
            if (INPUT_SIZE + i < publicInputs.length) {{
                expectedOutputs[i] = publicInputs[INPUT_SIZE + i];
            }}
        }}
        
        // Call the underlying EZKL verifier
        (bool verifySuccess, bytes memory result) = underlyingVerifier.call(proofData);
        
        if (!verifySuccess) {{
            emit VerificationFailed("Underlying verifier call failed");
            return (false, new uint256[](0));
        }}
        
        // Decode the verification result
        bool verified;
        try this.decodeVerificationResult(result) returns (bool _verified) {{
            verified = _verified;
        }} catch {{
            emit VerificationFailed("Failed to decode verification result");
            return (false, new uint256[](0));
        }}
        
        if (!verified) {{
            emit VerificationFailed("Proof verification failed");
            return (false, new uint256[](0));
        }}
        
        // If verification passed, return the expected outputs
        emit ProofVerified(true, expectedOutputs);
        return (true, expectedOutputs);
    }}
    
    /**
     * Helper function to decode verification result
     * Made external to allow try/catch
     */
    function decodeVerificationResult(bytes memory result) external pure returns (bool) {{
        return abi.decode(result, (bool));
    }}
    
    /**
     * Direct verification without output extraction (for compatibility)
     */
    function verifyProof(bytes calldata proofData) external returns (bool) {{
        (bool success, bytes memory result) = underlyingVerifier.call(proofData);
        if (!success) return false;
        
        try this.decodeVerificationResult(result) returns (bool verified) {{
            return verified;
        }} catch {{
            return false;
        }}
    }}
    
    /**
     * Get contract information
     */
    function getContractInfo() external view returns (
        address underlying,
        uint256 inputSize,
        uint256 outputSize,
        string memory modelName
    ) {{
        return (underlyingVerifier, INPUT_SIZE, OUTPUT_SIZE, "{model_name}");
    }}
}}"""
        
        return contract
        
    def generate_sequential_deployment_script(self, output_dir: str) -> str:
        """Generate deployment script for sequential chain"""
        
        script = f"""const hre = require("hardhat");
const fs = require("fs");

async function main() {{
    console.log("🚀 Deploying Sequential Proof Composition Chain");
    console.log("================================================================");
    console.log("⏰ Start time:", new Date().toISOString());
    
    const [deployer] = await hre.ethers.getSigners();
    console.log("👤 Deploying with account:", deployer.address);
    
    const balance = await deployer.provider.getBalance(deployer.address);
    console.log("💰 Account balance:", hre.ethers.formatEther(balance), "ETH");
    
    const network = await deployer.provider.getNetwork();
    console.log("🌐 Network:", network.name, "| Chain ID:", network.chainId);
    console.log("================================================================");
    
    // Step 1: Deploy underlying EZKL verifiers
    console.log("\\n📋 Deploying underlying EZKL verifiers...");
    const underlyingVerifiers = [];
    const stepNames = [];
    const inputSizes = [];
    const outputSizes = [];
    
"""
        
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            input_shape = model.get_input_shape()
            output_shape = model.get_output_shape()
            
            script += f"""    // Deploy step {i}: {model_name}
    console.log("Deploying step {i} underlying verifier: {model_name}");
    const Step{i}Verifier = await hre.ethers.getContractFactory("contracts/step{i:02d}_{model_name}Verifier.sol:Halo2Verifier");
    const step{i}Verifier = await Step{i}Verifier.deploy();
    await step{i}Verifier.waitForDeployment();
    
    const step{i}Address = await step{i}Verifier.getAddress();
    console.log("✅ Step {i} underlying verifier deployed:", step{i}Address);
    underlyingVerifiers.push(step{i}Address);
    stepNames.push("{model_name}");
    inputSizes.push({input_shape[0] if input_shape else 0});
    outputSizes.push({output_shape[0] if output_shape else 0});
    
"""
        
        script += f"""    // Step 2: Deploy enhanced verifiers
    console.log("\\n🔧 Deploying enhanced verifiers...");
    const enhancedVerifiers = [];
    
"""
        
        for i, position in enumerate(self.chain):
            model = position.model
            model_name = model.get_config()['name']
            
            script += f"""    // Deploy enhanced {model_name} verifier
    console.log("Deploying enhanced verifier for {model_name}");
    const Enhanced{model_name.title()}Verifier = await hre.ethers.getContractFactory("Enhanced{model_name.title()}Verifier");
    const enhanced{model_name.title()}Verifier = await Enhanced{model_name.title()}Verifier.deploy(underlyingVerifiers[{i}]);
    await enhanced{model_name.title()}Verifier.waitForDeployment();
    
    const enhanced{model_name.title()}Address = await enhanced{model_name.title()}Verifier.getAddress();
    console.log("✅ Enhanced {model_name} verifier deployed:", enhanced{model_name.title()}Address);
    enhancedVerifiers.push(enhanced{model_name.title()}Address);
    
"""
        
        script += f"""    // Step 3: Deploy sequential chain executor
    console.log("\\n⛓️  Deploying sequential chain executor...");
    const SequentialChainExecutor = await hre.ethers.getContractFactory("SequentialChainExecutor");
    const sequentialChainExecutor = await SequentialChainExecutor.deploy();
    await sequentialChainExecutor.waitForDeployment();
    
    const executorAddress = await sequentialChainExecutor.getAddress();
    console.log("✅ Sequential chain executor deployed:", executorAddress);
    
    // Step 4: Configure the chain
    console.log("\\n🔗 Configuring sequential chain...");
    const setChainTx = await sequentialChainExecutor.setChainSteps(
        enhancedVerifiers,
        stepNames,
        inputSizes,
        outputSizes
    );
    await setChainTx.wait();
    console.log("✅ Chain configured successfully");
    
    // Save deployment information
    const deploymentInfo = {{
        sequentialChainExecutor: executorAddress,
        underlyingVerifiers: underlyingVerifiers,
        enhancedVerifiers: enhancedVerifiers,
        stepNames: stepNames,
        inputSizes: inputSizes,
        outputSizes: outputSizes,
        chainLength: {len(self.chain)},
        timestamp: new Date().toISOString()
    }};
    
    fs.writeFileSync("sequential_deployment.json", JSON.stringify(deploymentInfo, null, 2));
    console.log("\\n📁 Deployment info saved to sequential_deployment.json");
    
    // Final summary
    console.log("\\n🎉 SEQUENTIAL DEPLOYMENT SUMMARY");
    console.log("================================================================");
    console.log("⛓️  Sequential Executor:", executorAddress);
    console.log("📊 Enhanced Verifiers:");
    for (let i = 0; i < enhancedVerifiers.length; i++) {{
        console.log(`   ${{i}}: ${{stepNames[i]}} -> ${{enhancedVerifiers[i]}}`);
    }}
    console.log("⏰ Completion time:", new Date().toISOString());
    console.log("🎯 Ready for sequential chain execution!");
    console.log("================================================================");
}}

main()
    .then(() => process.exit(0))
    .catch((error) => {{
        console.error("💥 Deployment failed:", error);
        console.error("🔍 Stack trace:", error.stack);
        process.exit(1);
    }});"""
        
        return script
        
    def generate_sequential_test_script(self) -> str:
        """Generate test script for sequential execution"""
        
        script = f"""const hre = require("hardhat");
const fs = require("fs");

async function main() {{
    console.log("🧪 Testing Sequential Proof Composition Chain");
    console.log("================================================================");
    
    // Load deployment info
    const deploymentInfo = JSON.parse(fs.readFileSync("sequential_deployment.json", "utf8"));
    console.log("📍 Sequential executor:", deploymentInfo.sequentialChainExecutor);
    
    // Get contract instance
    const sequentialExecutor = await hre.ethers.getContractAt(
        "SequentialChainExecutor", 
        deploymentInfo.sequentialChainExecutor
    );
    
    // Prepare test inputs
    console.log("\\n🎯 Preparing test data...");
    const initialInput = [/* Initial test input values */];
    const proofData = [
        /* Load actual proof data for each step */
    ];
    
    console.log("Initial input size:", initialInput.length);
    console.log("Proof data count:", proofData.length);
    
    try {{
        console.log("\\n⚡ Executing sequential chain...");
        const tx = await sequentialExecutor.executeSequentialChain(initialInput, proofData);
        const receipt = await tx.wait();
        
        console.log("📋 Transaction hash:", receipt.hash);
        console.log("⛽ Gas used:", receipt.gasUsed.toString());
        console.log("✅ Status:", receipt.status === 1 ? "SUCCESS" : "FAILED");
        
        // Parse events
        const events = receipt.logs;
        console.log("\\n📊 Execution Events:");
        
        for (const event of events) {{
            try {{
                const parsedEvent = sequentialExecutor.interface.parseLog(event);
                if (parsedEvent.name === "StepExecuted") {{
                    console.log(`   Step ${{parsedEvent.args.stepIndex}}: ${{parsedEvent.args.stepName}} - ${{parsedEvent.args.success ? "✅" : "❌"}}`);
                    if (parsedEvent.args.success) {{
                        console.log(`      Output: [${{parsedEvent.args.output.join(", ")}}]`);
                        console.log(`      Gas: ${{parsedEvent.args.gasUsed}}`);
                    }}
                }} else if (parsedEvent.name === "ChainCompleted") {{
                    console.log(`\\n🎉 Chain completed: ${{parsedEvent.args.success ? "✅ SUCCESS" : "❌ FAILED"}}`);
                    console.log(`   Final output: [${{parsedEvent.args.finalOutput.join(", ")}}]`);
                    console.log(`   Total gas: ${{parsedEvent.args.totalGasUsed}}`);
                }}
            }} catch (e) {{
                // Skip unparseable events
            }}
        }}
        
    }} catch (error) {{
        console.error("💥 Sequential execution failed:", error.message);
        console.error("🔍 Error details:", error);
    }}
}}

main()
    .then(() => process.exit(0))
    .catch((error) => {{
        console.error("💥 Test failed:", error);
        process.exit(1);
    }});"""
        
        return script