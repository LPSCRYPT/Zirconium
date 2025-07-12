// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Halo2VerifierBase
 * @dev Base contract containing shared constants and functions for all EZKL Halo2 verifiers
 */
abstract contract Halo2VerifierBase {
    
    // Shared curve constants
    uint256 internal constant DELTA = 4131629893567559867359510883348571134090853742863529169391034518566172092834;
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617; 

    // Shared memory pointer constants (common across all verifiers)
    uint256 internal constant VK_MPTR = 0x05a0;
    uint256 internal constant VK_DIGEST_MPTR = 0x05a0;
    uint256 internal constant NUM_INSTANCES_MPTR = 0x05c0;
    uint256 internal constant K_MPTR = 0x05e0;
    uint256 internal constant N_INV_MPTR = 0x0600;
    uint256 internal constant OMEGA_MPTR = 0x0620;
    uint256 internal constant OMEGA_INV_MPTR = 0x0640;
    uint256 internal constant OMEGA_INV_TO_L_MPTR = 0x0660;
    uint256 internal constant HAS_ACCUMULATOR_MPTR = 0x0680;
    uint256 internal constant ACC_OFFSET_MPTR = 0x06a0;
    uint256 internal constant NUM_ACC_LIMBS_MPTR = 0x06c0;
    uint256 internal constant NUM_ACC_LIMB_BITS_MPTR = 0x06e0;
    uint256 internal constant G1_X_MPTR = 0x0700;
    uint256 internal constant G1_Y_MPTR = 0x0720;
    uint256 internal constant G2_X_1_MPTR = 0x0740;
    uint256 internal constant G2_X_2_MPTR = 0x0760;
    uint256 internal constant G2_Y_1_MPTR = 0x0780;
    uint256 internal constant G2_Y_2_MPTR = 0x07a0;
    uint256 internal constant NEG_S_G2_X_1_MPTR = 0x07c0;
    uint256 internal constant NEG_S_G2_X_2_MPTR = 0x07e0;
    uint256 internal constant NEG_S_G2_Y_1_MPTR = 0x0800;
    uint256 internal constant NEG_S_G2_Y_2_MPTR = 0x0820;

    // Quotient polynomial memory pointers (shared)
    uint256 internal constant FIRST_QUOTIENT_X_CPTR = 0x0f24;
    uint256 internal constant LAST_QUOTIENT_X_CPTR = 0x10e4;

    /**
     * @dev Main verification function - must be implemented by child contracts with their specific memory layout
     */
    function verifyProof(
        bytes calldata proof,
        uint256[] calldata instances
    ) public virtual returns (bool);

    /**
     * @dev Shared assembly functions used by all verifiers
     * These are embedded in assembly blocks in child contracts but represent identical logic
     */
    
    // Note: The actual shared assembly functions (read_ec_point, squeeze_challenge, etc.) 
    // cannot be extracted from assembly blocks due to Solidity limitations.
    // Each verifier must still contain the full assembly implementation.
    // This base contract primarily saves space through shared constants.
}