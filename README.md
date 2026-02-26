Here is a GitHub-ready README in a single Markdown text box.
All math has been rewritten in plain text so it renders correctly on GitHub (no LaTeX).

⸻


# Quantum Platform  
### Neural–Symbolic Quantum Compilation & Research Environment

A full-stack experimental platform for quantum compiler research, learned intermediate representations (LIR), neural-guided optimization, and quantum algorithm discovery.

This project implements and extends the ideas proposed in:

“Learned Intermediate Representations in Quantum Compilers”  
João V. Pansani Relvas  
Instituto Superior Técnico

The repository serves as the practical research infrastructure for developing and validating a continuous latent representation for quantum circuits integrated into a symbolic compiler pipeline.

---

# Vision

Modern quantum compilers rely heavily on symbolic, rule-based rewriting over gate-level intermediate representations (IRs). While effective, these approaches are inherently local and limited by handcrafted heuristics.

This platform explores a different paradigm:

Hybrid neural–symbolic quantum compilation

Instead of optimizing circuits purely through deterministic rewrite rules, we introduce a learned continuous latent representation that guides optimization decisions while preserving strict symbolic validity guarantees.

---

# System Architecture

Quantum Language
    ↓
Front-End Parser
    ↓
Graph-Based Intermediate Representation (IR)
    ↓
-------------------------------------------------
|  Baseline Symbolic Optimization Pipeline      |
|  Neural-Guided LIR Optimization Pipeline      |
-------------------------------------------------
    ↓
Hardware Mapping Layer
    ↓
Circuit-Level Quantum Simulator

The system supports direct comparison between:

1) Pure symbolic rule-based compilation  
2) Neural-guided latent-space optimization  

under identical execution and validation conditions.

---

# Core Components

## 1. Quantum Compiler

A modular compiler stack including:

- High-level quantum language
- Graph-based structural IR
- Gate-level IR
- Hardware-level IR
- Optimization IR
- Continuous Latent IR (Learned IR)

Two compilation modes:

Symbolic Mode:
- Deterministic rule-based rewriting
- Gate cancellation
- Commutation analysis
- Hardware-aware mapping

Neural Mode:
- Learned ranking of candidate transformations
- Latent structural embeddings guide optimization
- Symbolic layer guarantees structural validity

---

## 2. Learned Intermediate Representation (LIR)

We define an encoder:

phi(C) -> R^d

Where:
- C = quantum circuit represented as a DAG
- phi(C) = d-dimensional continuous embedding

This embedding captures:

- Gate composition and frequency
- Dependency structure
- Circuit depth characteristics
- Parameterized gate information
- Optional hardware topology information

Optimization becomes:

R(C) --O--> R(C')

Where:
- O is a learnable operator
- Symbolic validation reconstructs C'

The symbolic layer enforces:

- Acyclicity
- Gate-set compliance
- Parameter consistency
- Qubit index validity

The neural component guides decisions.  
The symbolic component guarantees correctness.

---

## 3. Neural Optimization Modules

### Compiler Optimization Network
- Predicts cost impact of candidate rewrites
- Prioritizes transformations reducing:
  - Two-qubit gate count
  - Circuit depth
  - Routing overhead

### Hardware Optimization Network
- Learns hardware-aware gate decompositions
- Adapts to connectivity constraints

### Qubit Control & Error-Correction Module
- Experimental exploration of noise-aware compilation
- Adaptive routing under hardware constraints

### Neural Circuit Discovery Module
- Latent-space exploration
- Alternative decompositions
- Structural algorithm mutations

---

## 4. Quantum Simulator

Since physical hardware is not available, the platform includes a circuit-level simulator supporting:

- Statevector evolution
- Fidelity computation
- Structural metric evaluation
- Noise modeling (extensible)

Functional equivalence is verified via:

F = | <psi_baseline | psi_neural> |^2

Current limit:
- Approximately 8 qubits (full statevector simulation)

Future extensions:
- Tensor-network simulation
- Approximate simulation
- External hardware execution

---

## 5. Quantum Machine Learning Engine

Infrastructure for:

- Graph neural networks over circuit DAGs
- Supervised cost prediction
- Reinforcement learning over compilation trajectories
- Hardware-conditioned embeddings

Compilation can be modeled as:

C_(t+1) = T(C_t, a_t)

Policy:

pi(a_t | z_t)

Where:
- z_t = phi(C_t)

This enables sequential decision-based compilation.

---

# Research Objectives

1. Structural Compression  
   Can learned embeddings reduce two-qubit gate count beyond handcrafted rules?

2. Global Circuit Equivalence Detection  
   Can latent space capture non-local structural redundancies?

3. Hardware-Aware Embedding  
   Can topology and noise constraints be internalized?

4. Neural-Guided Routing  
   Can learned ranking improve qubit mapping?

5. Reinforcement-Based Compilation  
   Can compilation be treated as a policy-learning problem?

6. Autonomous Algorithm Discovery  
   Can latent exploration yield novel circuit structures?

---

# Evaluation Metrics

Primary objective:

J(C) = alpha * N_2q(C)

Where:
- N_2q(C) = number of two-qubit gates

Future objective:

J(C) = alpha * N_2q(C)  
      + beta * Depth(C)  
      + gamma * TotalGateCount(C)

Benchmarks:

- Grover circuits
- QFT circuits
- Random circuits

All neural results are compared directly to the symbolic baseline.

---

# Current Limitations

- Statevector simulation scalability
- Small-qubit regime
- Early-stage neural training
- No access to real calibration data
- No formal optimality guarantees

The neural component is a heuristic approximator, not an exact optimizer.

---

# Long-Term Vision

Move quantum compilation from:

Static symbolic rewriting

to

Hybrid neural–symbolic continuous optimization

Enabling:

- Noise-adaptive compilation
- Hardware-calibration-aware embeddings
- Pulse-level integration
- Latent-guided algorithm synthesis
- Self-improving compilation systems

---

# Status

Active research & development.

The architecture will evolve as experiments from the LIR framework are implemented and validated.

This repository is both:

- A compiler engineering project  
- A research laboratory for neural-guided quantum compilation  