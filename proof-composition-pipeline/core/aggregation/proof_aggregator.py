#!/usr/bin/env python3
"""
Proof Aggregation System

Implements advanced proof aggregation techniques for large chains:
- Merkle tree aggregation for scalable verification
- Recursive proof composition
- Batch verification optimization
- Proof compression and optimization
"""

import os
import sys
import json
import hashlib
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class AggregationStrategy(Enum):
    """Different aggregation strategies"""
    MERKLE_TREE = "merkle_tree"
    RECURSIVE = "recursive"
    BATCH = "batch"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"

@dataclass
class ProofMetadata:
    """Metadata for individual proofs"""
    model_id: str
    position: int
    proof_size: int
    verification_time: float
    gas_cost: int
    proof_hash: str

@dataclass
class AggregatedProof:
    """Result of proof aggregation"""
    aggregation_id: str
    strategy: AggregationStrategy
    constituent_proofs: List[ProofMetadata]
    aggregated_proof_data: Dict[str, Any]
    verification_contract: str
    metadata: Dict[str, Any]

class MerkleTreeAggregator:
    """Merkle tree-based proof aggregation"""
    
    def __init__(self):
        self.tree_levels = []
        
    def _hash_proof(self, proof_data: Dict[str, Any]) -> str:
        """Create a hash of proof data"""
        proof_str = json.dumps(proof_data, sort_keys=True)
        return hashlib.sha256(proof_str.encode()).hexdigest()
        
    def _build_merkle_tree(self, proof_hashes: List[str]) -> List[List[str]]:
        """Build a Merkle tree from proof hashes"""
        if not proof_hashes:
            return []
            
        # Pad to power of 2
        size = len(proof_hashes)
        next_power = 2 ** math.ceil(math.log2(size))
        padded_hashes = proof_hashes + ['0' * 64] * (next_power - size)
        
        levels = [padded_hashes]
        current_level = padded_hashes
        
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else '0' * 64
                combined = hashlib.sha256((left + right).encode()).hexdigest()
                next_level.append(combined)
            levels.append(next_level)
            current_level = next_level
            
        return levels
        
    def aggregate_proofs(self, proofs: List[Dict[str, Any]], 
                        proof_metadata: List[ProofMetadata]) -> AggregatedProof:
        """Aggregate proofs using Merkle tree"""
        
        # Create hashes for all proofs
        proof_hashes = [self._hash_proof(proof) for proof in proofs]
        
        # Build Merkle tree
        tree_levels = self._build_merkle_tree(proof_hashes)
        root_hash = tree_levels[-1][0] if tree_levels else None
        
        # Create aggregated proof structure
        aggregated_proof_data = {
            "merkle_root": root_hash,
            "tree_levels": tree_levels,
            "leaf_hashes": proof_hashes,
            "original_proofs": proofs,
            "verification_strategy": "merkle_inclusion"
        }
        
        # Generate verification contract
        verification_contract = self._generate_merkle_verifier_contract(
            root_hash, len(proofs), proof_metadata
        )
        
        # Calculate metadata
        total_gas = sum(pm.gas_cost for pm in proof_metadata)
        total_size = sum(pm.proof_size for pm in proof_metadata)
        
        metadata = {
            "total_proofs": len(proofs),
            "merkle_depth": len(tree_levels) - 1,
            "compression_ratio": total_size / len(json.dumps(aggregated_proof_data)),
            "estimated_gas_savings": max(0, total_gas - (50000 + 5000 * len(proofs))),
            "verification_complexity": f"O(log {len(proofs)})"
        }
        
        return AggregatedProof(
            aggregation_id=f"merkle_{root_hash[:16]}",
            strategy=AggregationStrategy.MERKLE_TREE,
            constituent_proofs=proof_metadata,
            aggregated_proof_data=aggregated_proof_data,
            verification_contract=verification_contract,
            metadata=metadata
        )
        
    def _generate_merkle_verifier_contract(self, root_hash: str, 
                                         proof_count: int,
                                         proof_metadata: List[ProofMetadata]) -> str:
        """Generate Solidity contract for Merkle-based verification"""
        
        contract = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * Merkle Tree Proof Aggregation Verifier
 * 
 * Verifies {proof_count} aggregated proofs using Merkle tree inclusion.
 * Root hash: {root_hash}
 */

contract MerkleAggregatedVerifier {{
    
    bytes32 public constant MERKLE_ROOT = 0x{root_hash};
    uint256 public constant PROOF_COUNT = {proof_count};
    
    struct MerkleProof {{
        bytes32[] siblings;
        uint256 index;
        bytes32 leaf;
    }}
    
    event ProofVerified(bytes32 indexed leafHash, uint256 index, bool success);
    event BatchVerified(uint256 proofCount, bool allValid);
    
    function verifyMerkleInclusion(MerkleProof memory proof) public pure returns (bool) {{
        bytes32 computedHash = proof.leaf;
        uint256 index = proof.index;
        
        for (uint256 i = 0; i < proof.siblings.length; i++) {{
            bytes32 sibling = proof.siblings[i];
            
            if (index % 2 == 0) {{
                computedHash = keccak256(abi.encodePacked(computedHash, sibling));
            }} else {{
                computedHash = keccak256(abi.encodePacked(sibling, computedHash));
            }}
            
            index = index / 2;
        }}
        
        return computedHash == MERKLE_ROOT;
    }}
    
    function verifyBatch(MerkleProof[] memory proofs) external returns (bool) {{
        require(proofs.length <= PROOF_COUNT, "Too many proofs");
        
        bool allValid = true;
        
        for (uint256 i = 0; i < proofs.length; i++) {{
            bool isValid = verifyMerkleInclusion(proofs[i]);
            emit ProofVerified(proofs[i].leaf, proofs[i].index, isValid);
            
            if (!isValid) {{
                allValid = false;
            }}
        }}
        
        emit BatchVerified(proofs.length, allValid);
        return allValid;
    }}
    
    function getAggregationInfo() external pure returns (bytes32, uint256) {{
        return (MERKLE_ROOT, PROOF_COUNT);
    }}
}}"""
        
        return contract

class RecursiveAggregator:
    """Recursive proof composition for unlimited scaling"""
    
    def __init__(self, max_batch_size: int = 8):
        self.max_batch_size = max_batch_size
        
    def aggregate_proofs(self, proofs: List[Dict[str, Any]], 
                        proof_metadata: List[ProofMetadata]) -> AggregatedProof:
        """Recursively aggregate proofs"""
        
        if len(proofs) <= self.max_batch_size:
            # Direct aggregation for small batches
            return self._aggregate_batch(proofs, proof_metadata)
        
        # Recursive aggregation for large sets
        batches = []
        batch_metadata = []
        
        for i in range(0, len(proofs), self.max_batch_size):
            batch_proofs = proofs[i:i + self.max_batch_size]
            batch_meta = proof_metadata[i:i + self.max_batch_size]
            
            batch_result = self._aggregate_batch(batch_proofs, batch_meta)
            batches.append(batch_result.aggregated_proof_data)
            
            # Create metadata for the aggregated batch
            batch_metadata.append(ProofMetadata(
                model_id=f"batch_{i // self.max_batch_size}",
                position=i // self.max_batch_size,
                proof_size=len(json.dumps(batch_result.aggregated_proof_data)),
                verification_time=0.0,
                gas_cost=100000,  # Estimated
                proof_hash=hashlib.sha256(
                    json.dumps(batch_result.aggregated_proof_data).encode()
                ).hexdigest()
            ))
        
        # Recursively aggregate the batches
        return self.aggregate_proofs(batches, batch_metadata)
        
    def _aggregate_batch(self, proofs: List[Dict[str, Any]], 
                        proof_metadata: List[ProofMetadata]) -> AggregatedProof:
        """Aggregate a single batch of proofs"""
        
        aggregated_proof_data = {
            "batch_proofs": proofs,
            "batch_size": len(proofs),
            "verification_strategy": "recursive_batch",
            "aggregation_level": 0
        }
        
        verification_contract = self._generate_recursive_verifier_contract(len(proofs))
        
        metadata = {
            "batch_size": len(proofs),
            "is_leaf_batch": True,
            "recursion_depth": 1
        }
        
        return AggregatedProof(
            aggregation_id=f"recursive_batch_{len(proofs)}",
            strategy=AggregationStrategy.RECURSIVE,
            constituent_proofs=proof_metadata,
            aggregated_proof_data=aggregated_proof_data,
            verification_contract=verification_contract,
            metadata=metadata
        )
        
    def _generate_recursive_verifier_contract(self, batch_size: int) -> str:
        """Generate contract for recursive verification"""
        
        contract = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract RecursiveAggregatedVerifier {{
    
    uint256 public constant BATCH_SIZE = {batch_size};
    
    struct RecursiveProof {{
        bytes[] individualProofs;
        address[] verifierContracts;
        bool isLeafProof;
    }}
    
    function verifyRecursive(RecursiveProof memory proof) external returns (bool) {{
        require(proof.individualProofs.length == proof.verifierContracts.length, "Mismatched arrays");
        require(proof.individualProofs.length <= BATCH_SIZE, "Batch too large");
        
        bool allValid = true;
        
        for (uint256 i = 0; i < proof.individualProofs.length; i++) {{
            (bool success, bytes memory result) = proof.verifierContracts[i].call(
                proof.individualProofs[i]
            );
            
            if (!success || !abi.decode(result, (bool))) {{
                allValid = false;
                break;
            }}
        }}
        
        return allValid;
    }}
}}"""
        
        return contract

class BatchVerificationOptimizer:
    """Optimize batch verification for gas efficiency"""
    
    def __init__(self):
        self.gas_estimates = {
            "base_cost": 21000,
            "per_proof_cost": 15000,
            "merkle_verification": 5000,
            "batch_overhead": 2000
        }
        
    def optimize_batch_sizes(self, proofs: List[Dict[str, Any]], 
                           max_gas_per_batch: int = 8000000) -> List[List[int]]:
        """Optimize batch sizes for gas efficiency"""
        
        # Estimate gas cost per proof
        proof_costs = []
        for i, proof in enumerate(proofs):
            estimated_cost = self.gas_estimates["per_proof_cost"]
            # Adjust based on proof complexity
            if len(json.dumps(proof)) > 10000:
                estimated_cost *= 1.5
            proof_costs.append(estimated_cost)
        
        # Create optimal batches
        batches = []
        current_batch = []
        current_gas = self.gas_estimates["base_cost"]
        
        for i, cost in enumerate(proof_costs):
            if current_gas + cost + self.gas_estimates["batch_overhead"] <= max_gas_per_batch:
                current_batch.append(i)
                current_gas += cost
            else:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [i]
                current_gas = self.gas_estimates["base_cost"] + cost
        
        if current_batch:
            batches.append(current_batch)
            
        return batches
        
    def generate_optimized_verifier(self, batch_indices: List[List[int]], 
                                  total_proofs: int) -> str:
        """Generate optimized batch verifier contract"""
        
        contract = f"""// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract OptimizedBatchVerifier {{
    
    uint256 public constant TOTAL_PROOFS = {total_proofs};
    uint256 public constant BATCH_COUNT = {len(batch_indices)};
    
    struct BatchProof {{
        uint256 batchIndex;
        bytes[] proofs;
        address[] verifiers;
    }}
    
    mapping(uint256 => bool) public batchVerified;
    uint256 public verifiedBatchCount;
    
    event BatchVerified(uint256 indexed batchIndex, uint256 proofCount, bool success);
    event AllBatchesVerified(bool success);
    
    function verifyBatch(BatchProof memory batch) external returns (bool) {{
        require(batch.batchIndex < BATCH_COUNT, "Invalid batch index");
        require(!batchVerified[batch.batchIndex], "Batch already verified");
        
        bool allValid = true;
        
        for (uint256 i = 0; i < batch.proofs.length; i++) {{
            (bool success, bytes memory result) = batch.verifiers[i].call(batch.proofs[i]);
            
            if (!success || !abi.decode(result, (bool))) {{
                allValid = false;
                break;
            }}
        }}
        
        batchVerified[batch.batchIndex] = allValid;
        if (allValid) {{
            verifiedBatchCount++;
        }}
        
        emit BatchVerified(batch.batchIndex, batch.proofs.length, allValid);
        
        if (verifiedBatchCount == BATCH_COUNT) {{
            emit AllBatchesVerified(true);
        }}
        
        return allValid;
    }}
    
    function isFullyVerified() external view returns (bool) {{
        return verifiedBatchCount == BATCH_COUNT;
    }}
    
    function getBatchStatus(uint256 batchIndex) external view returns (bool) {{
        return batchVerified[batchIndex];
    }}
}}"""
        
        return contract

class ProofAggregationEngine:
    """Main engine for proof aggregation"""
    
    def __init__(self):
        self.merkle_aggregator = MerkleTreeAggregator()
        self.recursive_aggregator = RecursiveAggregator()
        self.batch_optimizer = BatchVerificationOptimizer()
        
    def choose_optimal_strategy(self, proof_count: int, 
                              total_gas_budget: int = 8000000) -> AggregationStrategy:
        """Choose the optimal aggregation strategy"""
        
        if proof_count <= 4:
            return AggregationStrategy.SEQUENTIAL
        elif proof_count <= 16:
            return AggregationStrategy.MERKLE_TREE
        elif proof_count <= 64:
            return AggregationStrategy.BATCH
        else:
            return AggregationStrategy.RECURSIVE
            
    def aggregate(self, proofs: List[Dict[str, Any]], 
                 proof_metadata: List[ProofMetadata],
                 strategy: Optional[AggregationStrategy] = None) -> AggregatedProof:
        """Aggregate proofs using the specified or optimal strategy"""
        
        if strategy is None:
            strategy = self.choose_optimal_strategy(len(proofs))
            
        if strategy == AggregationStrategy.MERKLE_TREE:
            return self.merkle_aggregator.aggregate_proofs(proofs, proof_metadata)
        elif strategy == AggregationStrategy.RECURSIVE:
            return self.recursive_aggregator.aggregate_proofs(proofs, proof_metadata)
        elif strategy == AggregationStrategy.BATCH:
            return self._batch_aggregate(proofs, proof_metadata)
        else:
            return self._sequential_aggregate(proofs, proof_metadata)
            
    def _batch_aggregate(self, proofs: List[Dict[str, Any]], 
                        proof_metadata: List[ProofMetadata]) -> AggregatedProof:
        """Batch aggregation with optimization"""
        
        batch_indices = self.batch_optimizer.optimize_batch_sizes(proofs)
        verification_contract = self.batch_optimizer.generate_optimized_verifier(
            batch_indices, len(proofs)
        )
        
        aggregated_proof_data = {
            "batch_indices": batch_indices,
            "original_proofs": proofs,
            "verification_strategy": "optimized_batch"
        }
        
        metadata = {
            "batch_count": len(batch_indices),
            "optimization_applied": True,
            "estimated_gas_savings": sum(pm.gas_cost for pm in proof_metadata) * 0.3
        }
        
        return AggregatedProof(
            aggregation_id=f"batch_{len(proofs)}_{len(batch_indices)}",
            strategy=AggregationStrategy.BATCH,
            constituent_proofs=proof_metadata,
            aggregated_proof_data=aggregated_proof_data,
            verification_contract=verification_contract,
            metadata=metadata
        )
        
    def _sequential_aggregate(self, proofs: List[Dict[str, Any]], 
                            proof_metadata: List[ProofMetadata]) -> AggregatedProof:
        """Simple sequential aggregation for small chains"""
        
        aggregated_proof_data = {
            "sequential_proofs": proofs,
            "verification_strategy": "sequential"
        }
        
        verification_contract = """// Sequential verification - use individual verifiers"""
        
        metadata = {
            "is_optimized": False,
            "verification_order": "sequential"
        }
        
        return AggregatedProof(
            aggregation_id=f"sequential_{len(proofs)}",
            strategy=AggregationStrategy.SEQUENTIAL,
            constituent_proofs=proof_metadata,
            aggregated_proof_data=aggregated_proof_data,
            verification_contract=verification_contract,
            metadata=metadata
        )