// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title BaseEZKLVerifier
 * @dev Base contract for EZKL proof verifiers with common functionality
 * @notice Provides shared verification tracking and interface compliance
 */
abstract contract BaseEZKLVerifier {
    
    // Verification tracking
    uint256 public totalVerifications;
    mapping(bytes32 => bool) public verifiedProofs;
    
    // Events
    event ProofVerified(bytes32 indexed proofHash, address indexed verifier, bool success);
    
    /**
     * @dev Get the architecture name - must be implemented by child contracts
     */
    function getArchitecture() external pure virtual returns (string memory);
    
    /**
     * @dev Get total number of verifications performed
     */
    function getTotalVerifications() external view returns (uint256) {
        return totalVerifications;
    }
    
    /**
     * @dev Main verification function - calls the real EZKL verifyProof with tracking
     * @param proof The EZKL proof bytes
     * @param publicInputs The public inputs from EZKL
     * @return success True if proof is valid
     */
    function verify(bytes calldata proof, uint256[] calldata publicInputs) external returns (bool) {
        // Generate proof hash for tracking
        bytes32 proofHash = keccak256(abi.encodePacked(proof, publicInputs, address(this)));
        
        // Check if already verified
        if (verifiedProofs[proofHash]) {
            emit ProofVerified(proofHash, address(this), true);
            return true;
        }
        
        // Call the real EZKL verification function
        bool isValid = _verifyProof(proof, publicInputs);
        
        // Update tracking if valid
        if (isValid) {
            totalVerifications++;
            verifiedProofs[proofHash] = true;
        }
        
        emit ProofVerified(proofHash, address(this), isValid);
        return isValid;
    }
    
    /**
     * @dev Internal verification function - must be implemented by child contracts
     * This should call the real EZKL verifyProof function
     */
    function _verifyProof(bytes calldata proof, uint256[] calldata publicInputs) internal virtual returns (bool);
}