
⸻

Quantum Environment

A Neural Quantum Compilation and Research Environment

A full-stack quantum computing research platform for compiler design, learned intermediate representations, neural optimization, and quantum algorithm discovery.

⸻

Overview

This repository contains the development of a modular quantum computing platform designed to support:
	•	Quantum compiler research
	•	Learned Intermediate Representations (LIR)
	•	Neural-guided circuit optimization
	•	Quantum simulation
	•	Quantum machine learning experimentation
	•	Hardware-aware optimization research
	•	Autonomous circuit and algorithm discovery

The platform is not merely a simulator or compiler it is a controlled experimental environment for studying how machine learning can fundamentally alter quantum compilation and circuit optimization.

This project directly implements and extends the ideas proposed in:

“Learned Intermediate Representations in Quantum Compilers”
João V. Pansani Relvas
Instituto Superior Técnico

The preprint introduces a continuous latent representation of quantum circuits that enables global structural reasoning beyond rule-based gate rewriting.

This repository serves as the practical research infrastructure for that work.

⸻

Architecture Overview

The platform is structured as a layered neural–symbolic system:

Quantum Language
        ↓
Front-End Parser
        ↓
Graph-Based Intermediate Representation (IR)
        ↓
-----------------------------------------------
|  Baseline Symbolic Optimization Pipeline   |
|  Neural-Guided LIR Optimization Pipeline   |
-----------------------------------------------
        ↓
Hardware Mapping Layer
        ↓
Circuit-Level Quantum Simulator

The system allows direct comparison between:
	•	Pure rule-based compilation
	•	Neural-guided latent-space optimization

under identical execution and validation conditions.

⸻

Core Components

1. Quantum Compiler

A full compiler stack that includes:
	•	High-level quantum language
	•	Graph-based Intermediate Representation (IR)
	•	Gate-level IR
	•	Hardware-level IR
	•	Optimization IR
	•	Multi-stage transformation passes

The compiler supports two modes:
	•	Symbolic Mode - deterministic rule-based rewriting
	•	Neural Mode - LIR-guided optimization over the same symbolic backbone

This ensures structural correctness while enabling data-driven transformation ranking.

⸻

2. Learned Intermediate Representation (LIR)

This is the central research contribution.

Instead of representing circuits purely as symbolic DAGs, the compiler includes:

\phi : C \rightarrow \mathbb{R}^d

Where:
	•	C is a quantum circuit graph
	•	\phi(C) is a continuous latent embedding

In this latent space:
	•	Structural similarity becomes geometric proximity
	•	Optimization can be framed as learned transformation ranking
	•	Non-local equivalences can be detected
	•	Hardware constraints can be embedded directly into representations

Optimization becomes:

R(C) \xrightarrow{O} R(C')

where O is a learnable operator guiding transformation selection.

The symbolic layer enforces:
	•	Acyclicity
	•	Gate-set compliance
	•	Parameter validity
	•	Qubit consistency

Thus maintaining physical correctness.

⸻

3. Neural Optimization Modules

The platform integrates multiple neural subsystems:

a) Compiler Optimization Network
	•	Predicts structural cost impact of candidate rewrites
	•	Prioritizes transformations expected to reduce:
	•	Two-qubit gate count
	•	Circuit depth
	•	Hardware routing cost

b) Hardware Optimization Network
	•	Learns gate selection and decomposition strategies
	•	Adapts to hardware topology constraints
	•	May condition on noise models

c) Qubit Control & Error Correction Network
	•	Designed to explore learned strategies for:
	•	Noise-aware compilation
	•	Error mitigation
	•	Adaptive routing under hardware constraints

d) Neural Circuit Discovery Module
	•	Latent-space exploration for:
	•	Novel circuit variants
	•	Alternative decompositions
	•	Structure-preserving algorithm mutations

⸻

4. Quantum Simulator

Since no physical quantum hardware is available, the platform includes a circuit-level simulator supporting:
	•	Statevector evolution
	•	Fidelity computation
	•	Structural metric evaluation
	•	Noise modeling (extensible)

Functional equivalence is verified via:

F = |\langle \psi_{baseline} | \psi_{neural} \rangle|^2

Currently supports up to ~8 qubits under full statevector simulation.

Future extensions:
	•	Tensor-network simulation
	•	Approximate simulation
	•	External hardware execution

⸻

5. Quantum Machine Learning Engine

The platform includes infrastructure for:
	•	Graph neural networks over circuit DAGs
	•	Latent embedding training
	•	Supervised cost prediction
	•	Reinforcement learning over compilation trajectories
	•	Hardware-conditioned embeddings

This allows treating compilation as:

C_{t+1} = T(C_t, a_t)

with policies:

\pi(a_t | z_t)

where z_t = \phi(C_t).

⸻

6. Multi-Level Intermediate Representations

The platform supports layered IRs:
	•	High-level semantic IR
	•	Graph-based structural IR
	•	Gate-level IR
	•	Hardware-mapped IR
	•	Optimization IR
	•	Latent IR (continuous)

Unlike frameworks such as:
	•	Qiskit
	•	Cirq
	•	t|ket⟩

this system explicitly integrates a continuous learned layer into the compilation stack rather than relying exclusively on rule-based rewriting.

⸻

Research Goals

This platform is built to experimentally investigate:

1. Structural Compression

Can learned representations reduce two-qubit gate counts beyond heuristic rewriting?

2. Global Circuit Equivalence Detection

Can latent embeddings identify non-local structural redundancies?

3. Hardware-Aware Latent Conditioning

Can embeddings internalize topology and noise constraints?

4. Neural-Guided Routing

Can qubit mapping be improved via learned ranking of routing strategies?

5. Reinforcement-Based Compilation

Can compilation be framed as a sequential decision process with learned policies?

6. Autonomous Algorithm Discovery

Can latent space exploration yield new algorithmic patterns?

⸻

Evaluation Metrics

Current objective function:

J(C) = \alpha N_{2q}(C)

Primary metric:
	•	Two-qubit gate count

Future metrics:
	•	Depth
	•	Total gate count
	•	Noise-weighted cost
	•	Routing overhead
	•	Compilation time

Benchmark circuits include:
	•	Grover
	•	QFT
	•	Random circuits

All neural results are compared against baseline symbolic optimization under identical conditions.

⸻

Current Limitations
	•	Statevector simulation scaling
	•	Limited qubit count
	•	Early-stage neural training
	•	No access to real hardware calibration data
	•	No formal optimality guarantees (heuristic model)

⸻

Long-Term Vision

This project aims to move quantum compilation from:

Static symbolic rewriting

to

Hybrid neural–symbolic continuous optimization

Ultimately enabling:
	•	Noise-adaptive compilers
	•	Hardware-calibration-aware embeddings
	•	Pulse-level integration
	•	Latent-guided algorithm synthesis
	•	Self-improving compilation systems

⸻

Project Status

Active research & development.

This repository is under continuous architectural evolution as experiments from the LIR preprint are implemented, validated, and extended.

⸻