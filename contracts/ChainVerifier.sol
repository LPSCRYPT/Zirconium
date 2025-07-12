// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Chain Verifier for Proof Composition
 * 
 * Verifies a chain of 3 composed proofs.
 * Chain ID: simple_processing_chain
 * Description: Feature extraction followed by classification and decision making
 */

contract ChainVerifier {
    // Updated for compilation
    
    struct ProofStep {
        address verifierContract;
        bytes proofCalldata;
        string modelName;
        uint256 position;
    }
    
    event ChainVerified(string chainId, uint256 steps, bool success);
    
    string public constant CHAIN_ID = "simple_processing_chain";
    uint256 public constant CHAIN_LENGTH = 3;
    
    // Individual verifier contract addresses (to be set after deployment)
    address public step00Verifier; // feature_extractor
    address public step01Verifier; // classifier
    address public step02Verifier; // decision_maker
    
    constructor() {
        // Verifier addresses will be set after individual deployments
    }
    
    function setVerifierAddresses(address[] memory verifiers) external {
        require(verifiers.length == CHAIN_LENGTH, "Invalid verifier count");
        step00Verifier = verifiers[0];
        step01Verifier = verifiers[1];
        step02Verifier = verifiers[2];
    }
    
    function verifyChain(ProofStep[] memory steps) external returns (bool) {
        require(steps.length == CHAIN_LENGTH, "Invalid step count");
        
        bool allValid = true;
        
        // Verify each step in the chain
        for (uint256 i = 0; i < steps.length; i++) {
            ProofStep memory step = steps[i];
            
            // Call the verifier contract for this step
            (bool success, bytes memory result) = step.verifierContract.call(step.proofCalldata);
            
            if (!success) {
                allValid = false;
                break;
            }
            
            // Check if verification returned true
            bool verified = abi.decode(result, (bool));
            if (!verified) {
                allValid = false;
                break;
            }
        }
        
        emit ChainVerified(CHAIN_ID, steps.length, allValid);
        return allValid;
    }
    
    function getChainInfo() external pure returns (string memory, uint256) {
        return (CHAIN_ID, CHAIN_LENGTH);
    }
}