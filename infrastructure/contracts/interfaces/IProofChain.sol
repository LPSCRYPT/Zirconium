// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title IProofChain
 * @dev Interface for proof-chaining operations
 */
interface IProofChain {
    
    event ChainExecuted(
        bytes32 indexed executionId,
        address indexed executor,
        address[] verifiers,
        bool success
    );
    
    function executeChain(
        address[] calldata verifiers,
        bytes[] calldata proofs,
        uint256[] calldata initialInputs
    ) external returns (bool success, bytes32 executionId);
    
    function getChainResult(bytes32 executionId) external view returns (
        bool success,
        uint256 stepsCompleted,
        uint256[] memory finalOutputs
    );
}