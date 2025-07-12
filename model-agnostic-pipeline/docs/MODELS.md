# 🧠 Model Architecture Documentation

This document provides comprehensive information about the three neural network architectures implemented in the Zirconium proof-chaining framework: **RWKV**, **Mamba**, and **xLSTM**.

## 📊 Model Overview

The repository contains **6 models total** - 3 lightweight "simple" versions and 3 full-scale "complex" versions of each architecture, all trained using [Hugging Face Accelerate](https://huggingface.co/docs/accelerate/en/index).

### Model Size Comparison

| Model | Simple Version | Complex Version | Architecture |
|-------|----------------|-----------------|--------------|
| **RWKV** | 5.9 KB (3 nodes) | 19.1 MB (377 nodes) | Linear Attention |
| **Mamba** | 8.5 KB (3 nodes) | 7.2 MB (5,452 nodes) | State Space Model |
| **xLSTM** | 7.2 KB (3 nodes) | 13.0 MB (1,865 nodes) | Extended LSTM |

### Input/Output Specifications

**Simple Models:**
- **Input**: `[batch_size, 10]` - 10-dimensional feature vectors
- **Output**: `[batch_size, 69]` - 69-dimensional output vectors
- **Architecture**: Linear → ReLU → Linear (3 nodes each)

**Complex Models:**
- **Input**: `[batch_size, sequence_length]` - Variable sequence length
- **Output**: `[batch_size, sequence_length, 69]` - 69-dimensional hidden states per token
- **Architecture**: Full implementation with hundreds/thousands of nodes

---

## 🔄 RWKV: Receptance Weighted Key Value

### Architecture Overview
RWKV introduces a novel "linear attention" mechanism that allows the model to function like a recurrent network while maintaining transformer-style representations.

### Key Innovations
- **Linear Attention**: Replaces traditional softmax attention with recursive computation
- **Receptance Mechanism**: Uses Receptance (R), Time-first (u), and Time-decay (w) matrices
- **Recurrent Properties**: Enables processing sequences of any length despite fixed training context
- **Memory Efficiency**: Lower computational complexity than traditional attention

### Technical Details
- **Simple Version**: 3-node proof-of-concept (Linear→ReLU→Linear)
- **Complex Version**: 377-node full implementation with recursive state tracking
- **Innovation**: Computes attention recursively with exponential scaling and numerical stability

### Use Cases
- Language modeling and text generation
- Long sequence processing with memory efficiency
- Applications requiring recurrent-like behavior with transformer performance

---

## 🐍 Mamba: Selective State Space Model

### Architecture Overview
Mamba is a completely attention-free architecture based on selective structured state space models (SSMs), designed to address transformers' computational inefficiency with long sequences.

### Key Innovations
- **Attention-Free**: No attention mechanisms whatsoever
- **Selective Mechanism**: Content-based reasoning to focus on specific input parts
- **Hardware-Aware**: Parallel algorithm optimized for modern hardware
- **Scalability**: Can handle very long sequences efficiently

### Technical Details
- **Simple Version**: 3-node demonstration model
- **Complex Version**: 5,452-node full implementation (most complex in our suite)
- **Components**: H3 and gated MLP blocks (Mamba blocks)
- **Processing**: Selective scan mechanism with mixer layers

### Use Cases
- Language modeling with extremely long context
- Fast inference applications
- Memory-constrained environments requiring efficient sequence processing

---

## 🔢 xLSTM: Extended Long Short-Term Memory

### Architecture Overview
xLSTM represents a modern evolution of LSTM architecture, addressing traditional limitations through exponential gating and modified memory structures.

### Key Innovations (Based on 2024 NeurIPS Paper)
- **Exponential Gating**: New gating mechanism with normalization and stabilization
- **Dual Variants**:
  - **sLSTM**: Scalar memory with scalar updates and new memory mixing
  - **mLSTM**: Matrix memory with covariance update rule (fully parallelizable)
- **Residual Architecture**: xLSTM blocks stacked residually

### Technical Details
- **Simple Version**: 3-node basic implementation
- **Complex Version**: 1,865-node full architecture
- **Memory Revision**: Addresses LSTM's inability to revise stored information
- **Scaling**: Designed to work with billions of parameters

### Use Cases
- Large-scale language modeling
- Applications requiring long-term memory with revision capabilities
- Scenarios where LSTM interpretability is preferred over transformer complexity

---

## 🔗 Proof-Chaining Architecture

### Sequential Composition
The three models are designed to work together in a **proof-chaining pipeline**:

```
Input → RWKV → Mamba → xLSTM → Output
```

### Compositional Benefits
1. **RWKV**: Efficient initial processing with linear attention
2. **Mamba**: Selective state space modeling for content filtering
3. **xLSTM**: Final processing with advanced memory mechanisms

### Zero-Knowledge Properties
- Each model generates a cryptographic proof of its computation
- Proofs can be verified independently or as a chain
- Enables trustless AI agent collaboration

---

## 🛠 Training Details

### Training Framework
All models were trained using **Hugging Face Accelerate**, enabling:
- Distributed training across multiple devices
- Efficient memory management for large models
- Seamless ONNX export for EZKL compatibility

### Data Preprocessing
- **Input Format**: 10-dimensional normalized feature vectors
- **Value Range**: Approximately [-2, 1] (standardized)
- **Output Dimension**: 69-dimensional hidden states

### Export Process
Models were exported to ONNX format for EZKL compatibility:
- **Simple Models**: Optimized for fast proof generation (~1-2 seconds)
- **Complex Models**: Full-scale implementations for production use
- **Verification**: All models include cryptographic proof generation

---

## 🎯 Applications & Use Cases

### Immediate Applications
1. **Verifiable AI Pipelines**: Trustless composition of different AI models
2. **Decentralized Inference**: Proof-carrying computation across untrusted nodes
3. **Model Auditing**: Cryptographic verification of AI decision processes

### Research Implications
1. **zkML Advancement**: Pushing boundaries of what's possible with zero-knowledge machine learning
2. **Architecture Comparison**: Direct comparison of cutting-edge sequence modeling approaches
3. **Scalability Studies**: Understanding proof generation complexity across model sizes

### Future Directions
1. **Dynamic Composition**: Runtime selection of proof-chaining paths
2. **Cross-Domain Applications**: Extending beyond sequence modeling
3. **Optimization**: Reducing proof generation time for complex models

---

## 📈 Performance Characteristics

### Proof Generation Times
- **Simple Models**: ~1-2 seconds (verified working)
- **Complex Models**: Minutes to hours (depending on hardware)

### Proof Sizes
- **Typical Range**: 50-60 KB per proof
- **Verification**: Sub-second verification times
- **Composition**: Proofs can be chained without size explosion

### Hardware Requirements
- **Minimum**: 8GB RAM for simple models
- **Recommended**: 32GB+ RAM for complex models
- **Storage**: SRS files require ~1GB for proof generation

---

## 🔍 Technical Notes

### Model Validation
All models include comprehensive validation through:
- **Artifact Validation**: `src/simple_ezkl_test.py`
- **End-to-End Testing**: `src/run_existing_ezkl_demo.py`
- **Integration Testing**: `src/weather_prediction.py`

### EZKL Compatibility
- **Version**: EZKL 22.0.1
- **Settings**: Auto-generated for each model
- **Circuits**: Pre-compiled for simple models, dynamic for complex
- **Verification**: Real cryptographic proofs (no mock data)

### Future Development
The model suite provides a foundation for:
- Advanced proof-chaining research
- Comparative architecture studies
- Production zkML applications
- Decentralized AI systems

---

*For technical implementation details, see the `/ezkl_workspace/` directory containing all model artifacts, settings, and proof generation scripts.*