// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Enhanced Feature_Extractor Verifier
 * 
 * Step 0 in the sequential chain.
 * Input Shape: [10]
 * Output Shape: [8]
 * 
 * This contract both verifies the proof AND returns the computation output
 * for use by the next step in the chain.
 */

import "./step00_feature_extractorVerifier.sol";

contract EnhancedFeature_ExtractorVerifier {
    
    // Reference to the underlying EZKL verifier
    address public immutable underlyingVerifier;
    
    // Expected input/output sizes
    uint256 public constant INPUT_SIZE = 10;
    uint256 public constant OUTPUT_SIZE = 8;
    
    event ProofVerified(bool success, uint256[] output);
    event VerificationFailed(string reason);
    
    constructor(address _underlyingVerifier) {
        underlyingVerifier = _underlyingVerifier;
    }
    
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
    ) external returns (bool success, uint256[] memory outputs) {
        
        // Validate input size
        if (publicInputs.length < INPUT_SIZE) {
            emit VerificationFailed("Insufficient public inputs");
            return (false, new uint256[](0));
        }
        
        // Extract inputs and expected outputs from publicInputs
        // Format: [input_0, input_1, ..., input_n, output_0, output_1, ..., output_m]
        uint256[] memory inputs = new uint256[](INPUT_SIZE);
        uint256[] memory expectedOutputs = new uint256[](OUTPUT_SIZE);
        
        for (uint256 i = 0; i < INPUT_SIZE; i++) {
            inputs[i] = publicInputs[i];
        }
        
        for (uint256 i = 0; i < OUTPUT_SIZE; i++) {
            if (INPUT_SIZE + i < publicInputs.length) {
                expectedOutputs[i] = publicInputs[INPUT_SIZE + i];
            }
        }
        
        // Call the underlying EZKL verifier
        (bool verifySuccess, bytes memory result) = underlyingVerifier.call(proofData);
        
        if (!verifySuccess) {
            emit VerificationFailed("Underlying verifier call failed");
            return (false, new uint256[](0));
        }
        
        // Decode the verification result
        bool verified;
        try this.decodeVerificationResult(result) returns (bool _verified) {
            verified = _verified;
        } catch {
            emit VerificationFailed("Failed to decode verification result");
            return (false, new uint256[](0));
        }
        
        if (!verified) {
            emit VerificationFailed("Proof verification failed");
            return (false, new uint256[](0));
        }
        
        // If verification passed, return the expected outputs
        emit ProofVerified(true, expectedOutputs);
        return (true, expectedOutputs);
    }
    
    /**
     * Helper function to decode verification result
     * Made external to allow try/catch
     */
    function decodeVerificationResult(bytes memory result) external pure returns (bool) {
        return abi.decode(result, (bool));
    }
    
    /**
     * Direct verification without output extraction (for compatibility)
     */
    function verifyProof(bytes calldata proofData) external returns (bool) {
        (bool success, bytes memory result) = underlyingVerifier.call(proofData);
        if (!success) return false;
        
        try this.decodeVerificationResult(result) returns (bool verified) {
            return verified;
        } catch {
            return false;
        }
    }
    
    /**
     * Get contract information
     */
    function getContractInfo() external view returns (
        address underlying,
        uint256 inputSize,
        uint256 outputSize,
        string memory modelName
    ) {
        return (underlyingVerifier, INPUT_SIZE, OUTPUT_SIZE, "feature_extractor");
    }
}