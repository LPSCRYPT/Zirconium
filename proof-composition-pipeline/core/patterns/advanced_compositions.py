#!/usr/bin/env python3
"""
Advanced Composition Patterns

Implements sophisticated composition patterns beyond simple sequential chains:
- Parallel composition (models execute concurrently)
- Conditional composition (dynamic routing based on conditions)
- Tree composition (branching and merging)
- Loop composition (iterative processing)
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable
from enum import Enum
import asyncio
import concurrent.futures

# Add interfaces to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'interfaces'))
from composable_model import ComposableModelInterface, CompositionChain, ChainPosition

class CompositionPattern(Enum):
    """Types of composition patterns supported"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    TREE = "tree"
    LOOP = "loop"
    PIPELINE = "pipeline"

class CompositionNode(ABC):
    """Abstract base for all composition nodes"""
    
    def __init__(self, node_id: str, pattern: CompositionPattern):
        self.node_id = node_id
        self.pattern = pattern
        self.metadata = {}
        
    @abstractmethod
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute this composition node"""
        pass
        
    @abstractmethod
    def get_output_shape(self) -> List[int]:
        """Get the output shape of this node"""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """Validate this node's configuration"""
        pass

class ModelNode(CompositionNode):
    """Single model execution node"""
    
    def __init__(self, node_id: str, model: ComposableModelInterface):
        super().__init__(node_id, CompositionPattern.SEQUENTIAL)
        self.model = model
        
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute the model"""
        import torch
        
        # Transform input
        transformed_input = self.model.transform_input(input_data)
        
        # Run inference
        torch_model = self.model.get_model()
        input_tensor = torch.tensor(transformed_input, dtype=torch.float32)
        
        with torch.no_grad():
            output_tensor = torch_model(input_tensor)
            raw_output = output_tensor.numpy().tolist()
            
        # Transform output
        return self.model.transform_output(raw_output)
        
    def get_output_shape(self) -> List[int]:
        return self.model.get_output_shape()
        
    def validate(self) -> bool:
        return True

class ParallelNode(CompositionNode):
    """Parallel execution of multiple models"""
    
    def __init__(self, node_id: str, models: List[ComposableModelInterface], 
                 aggregation_fn: Callable = None):
        super().__init__(node_id, CompositionPattern.PARALLEL)
        self.models = models
        self.aggregation_fn = aggregation_fn or self._default_concatenate
        
    def _default_concatenate(self, outputs: List[List[List[float]]]) -> List[List[float]]:
        """Default aggregation: concatenate all outputs"""
        if not outputs:
            return []
            
        # Concatenate along feature dimension
        result = []
        for i in range(len(outputs[0])):  # For each sample
            concatenated_features = []
            for output in outputs:  # For each model output
                concatenated_features.extend(output[i])
            result.append(concatenated_features)
        return result
        
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute all models in parallel"""
        
        def execute_model(model):
            node = ModelNode(f"{self.node_id}_model_{model.get_config()['name']}", model)
            return node.execute(input_data)
            
        # Run models in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            future_to_model = {executor.submit(execute_model, model): model for model in self.models}
            outputs = []
            
            for future in concurrent.futures.as_completed(future_to_model):
                result = future.result()
                outputs.append(result)
                
        return self.aggregation_fn(outputs)
        
    def get_output_shape(self) -> List[int]:
        # For concatenation, sum all output shapes
        total_features = sum(model.get_output_shape()[0] for model in self.models)
        return [total_features]
        
    def validate(self) -> bool:
        if not self.models:
            return False
        return all(isinstance(model, ComposableModelInterface) for model in self.models)

class ConditionalNode(CompositionNode):
    """Conditional routing based on data or model outputs"""
    
    def __init__(self, node_id: str, 
                 condition_fn: Callable[[List[List[float]]], str],
                 branches: Dict[str, CompositionNode],
                 default_branch: str = None):
        super().__init__(node_id, CompositionPattern.CONDITIONAL)
        self.condition_fn = condition_fn
        self.branches = branches
        self.default_branch = default_branch or list(branches.keys())[0]
        
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute based on condition"""
        branch_key = self.condition_fn(input_data)
        
        if branch_key not in self.branches:
            branch_key = self.default_branch
            
        selected_branch = self.branches[branch_key]
        return selected_branch.execute(input_data)
        
    def get_output_shape(self) -> List[int]:
        # Return the shape of the default branch
        return self.branches[self.default_branch].get_output_shape()
        
    def validate(self) -> bool:
        return (len(self.branches) > 0 and 
                self.default_branch in self.branches and
                all(isinstance(node, CompositionNode) for node in self.branches.values()))

class TreeNode(CompositionNode):
    """Tree-like composition with branching and merging"""
    
    def __init__(self, node_id: str, 
                 splitter: CompositionNode,
                 branches: List[CompositionNode],
                 merger: CompositionNode):
        super().__init__(node_id, CompositionPattern.TREE)
        self.splitter = splitter
        self.branches = branches
        self.merger = merger
        
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute tree composition"""
        # Split the data
        split_data = self.splitter.execute(input_data)
        
        # Process each branch
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch.execute(split_data)
            branch_outputs.append(branch_output)
            
        # Merge the results
        # For simplicity, concatenate all branch outputs
        merged_input = []
        for i in range(len(branch_outputs[0])):  # For each sample
            merged_features = []
            for branch_output in branch_outputs:
                merged_features.extend(branch_output[i])
            merged_input.append(merged_features)
            
        return self.merger.execute(merged_input)
        
    def get_output_shape(self) -> List[int]:
        return self.merger.get_output_shape()
        
    def validate(self) -> bool:
        return (self.splitter is not None and 
                len(self.branches) > 0 and
                self.merger is not None)

class LoopNode(CompositionNode):
    """Iterative composition with loop condition"""
    
    def __init__(self, node_id: str,
                 body: CompositionNode,
                 condition_fn: Callable[[List[List[float]], int], bool],
                 max_iterations: int = 10):
        super().__init__(node_id, CompositionPattern.LOOP)
        self.body = body
        self.condition_fn = condition_fn
        self.max_iterations = max_iterations
        
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute loop composition"""
        current_data = input_data
        iteration = 0
        
        while (iteration < self.max_iterations and 
               self.condition_fn(current_data, iteration)):
            current_data = self.body.execute(current_data)
            iteration += 1
            
        return current_data
        
    def get_output_shape(self) -> List[int]:
        return self.body.get_output_shape()
        
    def validate(self) -> bool:
        return self.body is not None and self.max_iterations > 0

class AdvancedCompositionChain:
    """Enhanced composition chain supporting advanced patterns"""
    
    def __init__(self, chain_id: str, description: str = ""):
        self.chain_id = chain_id
        self.description = description
        self.root_node: Optional[CompositionNode] = None
        self.nodes: Dict[str, CompositionNode] = {}
        self.metadata = {
            "pattern_types": set(),
            "total_models": 0,
            "estimated_complexity": "low"
        }
        
    def set_root(self, node: CompositionNode) -> 'AdvancedCompositionChain':
        """Set the root node of the composition"""
        self.root_node = node
        self._analyze_composition()
        return self
        
    def add_node(self, node: CompositionNode) -> 'AdvancedCompositionChain':
        """Add a node to the composition"""
        self.nodes[node.node_id] = node
        return self
        
    def _analyze_composition(self):
        """Analyze the composition for metadata"""
        if not self.root_node:
            return
            
        def analyze_node(node: CompositionNode):
            self.metadata["pattern_types"].add(node.pattern.value)
            
            if isinstance(node, ModelNode):
                self.metadata["total_models"] += 1
            elif isinstance(node, ParallelNode):
                self.metadata["total_models"] += len(node.models)
            elif isinstance(node, TreeNode):
                analyze_node(node.splitter)
                for branch in node.branches:
                    analyze_node(branch)
                analyze_node(node.merger)
            elif isinstance(node, LoopNode):
                analyze_node(node.body)
            elif isinstance(node, ConditionalNode):
                for branch in node.branches.values():
                    analyze_node(branch)
                    
        analyze_node(self.root_node)
        
        # Estimate complexity
        if len(self.metadata["pattern_types"]) > 2:
            self.metadata["estimated_complexity"] = "high"
        elif "parallel" in self.metadata["pattern_types"] or "tree" in self.metadata["pattern_types"]:
            self.metadata["estimated_complexity"] = "medium"
            
    def execute(self, input_data: List[List[float]]) -> List[List[float]]:
        """Execute the entire composition"""
        if not self.root_node:
            raise ValueError("No root node set for composition")
            
        return self.root_node.execute(input_data)
        
    def validate(self) -> bool:
        """Validate the entire composition"""
        if not self.root_node:
            return False
            
        return self.root_node.validate()
        
    def get_complexity_report(self) -> Dict[str, Any]:
        """Get a report on the composition's complexity"""
        return {
            "chain_id": self.chain_id,
            "pattern_types": list(self.metadata["pattern_types"]),
            "total_models": self.metadata["total_models"],
            "estimated_complexity": self.metadata["estimated_complexity"],
            "supports_parallelism": "parallel" in self.metadata["pattern_types"],
            "has_conditionals": "conditional" in self.metadata["pattern_types"],
            "has_loops": "loop" in self.metadata["pattern_types"]
        }

# Builder pattern for easy composition creation
class CompositionBuilder:
    """Builder for creating advanced compositions"""
    
    def __init__(self, chain_id: str, description: str = ""):
        self.chain = AdvancedCompositionChain(chain_id, description)
        
    def sequential(self, models: List[ComposableModelInterface]) -> 'CompositionBuilder':
        """Create a sequential chain"""
        if len(models) == 1:
            node = ModelNode("root", models[0])
        else:
            # Create nested sequential structure
            current_node = ModelNode(f"model_0", models[0])
            for i, model in enumerate(models[1:], 1):
                next_node = ModelNode(f"model_{i}", model)
                # For now, just use the last model as root
                current_node = next_node
                
        self.chain.set_root(current_node)
        return self
        
    def parallel(self, models: List[ComposableModelInterface], 
                aggregation_fn: Callable = None) -> 'CompositionBuilder':
        """Create a parallel composition"""
        node = ParallelNode("parallel_root", models, aggregation_fn)
        self.chain.set_root(node)
        return self
        
    def conditional(self, condition_fn: Callable, 
                   branches: Dict[str, ComposableModelInterface],
                   default_branch: str = None) -> 'CompositionBuilder':
        """Create a conditional composition"""
        branch_nodes = {key: ModelNode(f"branch_{key}", model) 
                       for key, model in branches.items()}
        node = ConditionalNode("conditional_root", condition_fn, branch_nodes, default_branch)
        self.chain.set_root(node)
        return self
        
    def build(self) -> AdvancedCompositionChain:
        """Build the final composition"""
        if not self.chain.validate():
            raise ValueError("Invalid composition configuration")
        return self.chain

# Example condition functions
class ConditionFunctions:
    """Common condition functions for conditional routing"""
    
    @staticmethod
    def input_magnitude_based(threshold: float = 0.5):
        """Route based on input magnitude"""
        def condition(data: List[List[float]]) -> str:
            avg_magnitude = sum(abs(x) for sample in data for x in sample) / sum(len(sample) for sample in data)
            return "high" if avg_magnitude > threshold else "low"
        return condition
        
    @staticmethod
    def feature_based(feature_index: int, threshold: float = 0.0):
        """Route based on specific feature value"""
        def condition(data: List[List[float]]) -> str:
            if data and len(data[0]) > feature_index:
                return "positive" if data[0][feature_index] > threshold else "negative"
            return "default"
        return condition
        
    @staticmethod
    def data_size_based(size_threshold: int = 100):
        """Route based on data size"""
        def condition(data: List[List[float]]) -> str:
            total_features = sum(len(sample) for sample in data)
            return "large" if total_features > size_threshold else "small"
        return condition

# Example loop conditions
class LoopConditions:
    """Common loop conditions"""
    
    @staticmethod
    def convergence_based(tolerance: float = 0.01):
        """Continue until output converges"""
        previous_output = None
        
        def condition(data: List[List[float]], iteration: int) -> bool:
            nonlocal previous_output
            if previous_output is None:
                previous_output = data
                return True
                
            # Check if output has converged
            total_diff = sum(abs(a - b) for sample_a, sample_b in zip(data, previous_output) 
                           for a, b in zip(sample_a, sample_b))
            previous_output = data
            return total_diff > tolerance
            
        return condition
        
    @staticmethod
    def output_magnitude_based(min_magnitude: float = 0.1):
        """Continue until output magnitude is below threshold"""
        def condition(data: List[List[float]], iteration: int) -> bool:
            avg_magnitude = sum(abs(x) for sample in data for x in sample) / sum(len(sample) for sample in data)
            return avg_magnitude > min_magnitude
        return condition