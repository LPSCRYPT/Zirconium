// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "../base/BaseEZKLVerifier.sol";
import "./MambaVerifier.sol";

/**
 * @title MambaVerifierWrapper
 * @dev Wrapper around real EZKL-generated MambaVerifier with inheritance structure
 */
contract MambaVerifierWrapper is BaseEZKLVerifier {
    
    // The real EZKL-generated verifier contract
    MambaVerifier private immutable ezkl_verifier;
    
    constructor() {
        ezkl_verifier = new MambaVerifier();
    }
    
    /**
     * @dev Get the architecture name for this verifier
     */
    function getArchitecture() external pure override returns (string memory) {
        return "Mamba";
    }
    
    /**
     * @dev Internal verification using real EZKL cryptographic verification
     * @param proof The EZKL proof bytes
     * @param publicInputs The public inputs from EZKL
     * @return valid True if the proof is cryptographically valid
     */
    function _verifyProof(bytes calldata proof, uint256[] calldata publicInputs) internal override returns (bool) {
        // Use low-level call as recommended by EZKL documentation
        bytes memory callData = abi.encodeWithSignature("verifyProof(bytes,uint256[])", proof, publicInputs);
        (bool success, bytes memory returnData) = address(ezkl_verifier).call(callData);
        
        if (!success) {
            return false;
        }
        
        return abi.decode(returnData, (bool));
    }
    
    /**
     * @dev Get the address of the underlying EZKL verifier
     */
    function getEZKLVerifier() external view returns (address) {
        return address(ezkl_verifier);
    }
}