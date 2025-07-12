// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IEZKLVerifier
 * @dev Interface for EZKL proof verifiers
 */
interface IEZKLVerifier {
    
    /**
     * @dev Verify an EZKL proof
     * @param proof The EZKL proof bytes
     * @param publicInputs The public inputs from EZKL
     * @return success True if proof is valid
     */
    function verify(bytes calldata proof, uint256[] calldata publicInputs) external returns (bool);
    
    /**
     * @dev Get the architecture name (RWKV, Mamba, xLSTM)
     */
    function getArchitecture() external pure returns (string memory);
    
    /**
     * @dev Get total number of verifications performed
     */
    function getTotalVerifications() external view returns (uint256);
    
    /**
     * @dev Event emitted when a proof is verified
     */
    event ProofVerified(bytes32 indexed proofHash, address indexed verifier, bool success);
}