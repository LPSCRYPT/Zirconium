// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Sequential Chain Executor
 * 
 * Executes a 3-step proof composition chain sequentially:
 * feature_extractor -> classifier -> decision_maker
 * 
 * Each step verifies a proof AND produces output for the next step.
 */

interface IEnhancedVerifier {
    function verifyAndExecute(
        bytes calldata proofData,
        uint256[] calldata publicInputs
    ) external returns (bool success, uint256[] memory outputs);
}

contract SequentialChainExecutor {
    
    struct ChainStep {
        address verifierContract;
        string stepName;
        uint256 expectedInputSize;
        uint256 expectedOutputSize;
    }
    
    struct ExecutionResult {
        bool success;
        uint256 totalSteps;
        uint256[][] intermediateOutputs;
        uint256[] finalOutput;
        uint256 totalGasUsed;
    }
    
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
    
    constructor() {
        // Initialize chain steps - addresses set after individual deployments
    }
    
    function setChainSteps(
        address[] memory verifierAddresses,
        string[] memory stepNames,
        uint256[] memory inputSizes,
        uint256[] memory outputSizes
    ) external {
        require(verifierAddresses.length == stepNames.length, "Mismatched arrays");
        require(verifierAddresses.length == inputSizes.length, "Mismatched arrays");
        require(verifierAddresses.length == outputSizes.length, "Mismatched arrays");
        
        delete chainSteps;
        
        for (uint256 i = 0; i < verifierAddresses.length; i++) {
            chainSteps.push(ChainStep({
                verifierContract: verifierAddresses[i],
                stepName: stepNames[i],
                expectedInputSize: inputSizes[i],
                expectedOutputSize: outputSizes[i]
            }));
        }
    }
    
    function executeSequentialChain(
        uint256[] memory initialInput,
        bytes[] memory proofData
    ) external returns (ExecutionResult memory) {
        
        require(chainSteps.length > 0, "Chain not configured");
        require(proofData.length == chainSteps.length, "Proof count mismatch");
        
        uint256 startGas = gasleft();
        uint256[][] memory intermediateOutputs = new uint256[][](chainSteps.length);
        uint256[] memory currentInput = initialInput;
        bool allSuccessful = true;
        
        // Execute each step sequentially
        for (uint256 i = 0; i < chainSteps.length; i++) {
            ChainStep memory step = chainSteps[i];
            
            // Verify input size matches expectation
            require(currentInput.length == step.expectedInputSize, 
                string(abi.encodePacked("Step ", toString(i), " input size mismatch")));
            
            uint256 stepStartGas = gasleft();
            
            // Call the enhanced verifier
            (bool stepSuccess, uint256[] memory stepOutput) = IEnhancedVerifier(step.verifierContract)
                .verifyAndExecute(proofData[i], currentInput);
            
            uint256 stepGasUsed = stepStartGas - gasleft();
            
            if (!stepSuccess) {
                allSuccessful = false;
                emit StepExecuted(i, step.stepName, false, new uint256[](0), stepGasUsed);
                break;
            }
            
            // Verify output size matches expectation
            require(stepOutput.length == step.expectedOutputSize,
                string(abi.encodePacked("Step ", toString(i), " output size mismatch")));
            
            // Store intermediate output
            intermediateOutputs[i] = stepOutput;
            
            // Output becomes input for next step
            currentInput = stepOutput;
            
            emit StepExecuted(i, step.stepName, true, stepOutput, stepGasUsed);
        }
        
        uint256 totalGasUsed = startGas - gasleft();
        
        ExecutionResult memory result = ExecutionResult({
            success: allSuccessful,
            totalSteps: chainSteps.length,
            intermediateOutputs: intermediateOutputs,
            finalOutput: allSuccessful ? currentInput : new uint256[](0),
            totalGasUsed: totalGasUsed
        });
        
        emit ChainCompleted(allSuccessful, chainSteps.length, result.finalOutput, totalGasUsed);
        
        return result;
    }
    
    function getChainInfo() external view returns (
        uint256 stepCount,
        string[] memory stepNames,
        address[] memory verifierAddresses
    ) {
        stepCount = chainSteps.length;
        stepNames = new string[](stepCount);
        verifierAddresses = new address[](stepCount);
        
        for (uint256 i = 0; i < stepCount; i++) {
            stepNames[i] = chainSteps[i].stepName;
            verifierAddresses[i] = chainSteps[i].verifierContract;
        }
    }
    
    function getStepInfo(uint256 stepIndex) external view returns (
        address verifierContract,
        string memory stepName,
        uint256 expectedInputSize,
        uint256 expectedOutputSize
    ) {
        require(stepIndex < chainSteps.length, "Invalid step index");
        ChainStep memory step = chainSteps[stepIndex];
        return (step.verifierContract, step.stepName, step.expectedInputSize, step.expectedOutputSize);
    }
    
    // Helper function to convert uint to string
    function toString(uint256 value) internal pure returns (string memory) {
        if (value == 0) return "0";
        
        uint256 temp = value;
        uint256 digits;
        while (temp != 0) {
            digits++;
            temp /= 10;
        }
        
        bytes memory buffer = new bytes(digits);
        while (value != 0) {
            digits -= 1;
            buffer[digits] = bytes1(uint8(48 + uint256(value % 10)));
            value /= 10;
        }
        
        return string(buffer);
    }
}