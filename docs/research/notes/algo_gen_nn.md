1. Role of the Research LLM (what it actually does)

This model is:

A transformer-based policy/value network that operates over quantum programs and proposes edits and structures to improve or discover quantum circuits, inside a search/RL environment.

It does not chat.
It sees programs, tasks, hardware, and outputs actions.

Core tasks:

Propose initial circuits for a given task + hardware

Propose edits to existing circuits in a search trajectory

Estimate value (future reward) of a given program

Optionally approximate metrics (fidelity, cost) to help prune search

2. Core Data Representations

Before tokens, define structured objects (conceptual types):

TaskSpec

defines the computational problem

e.g.: type, num_qubits, target_behavior, constraints

HardwareProfile

topology (connectivity graph)

native gate set

error/noise characteristics

Program / CircuitIR

list of gates, qubits, parameters, structure

possibly hierarchical (subroutines, blocks)

ProgramState

task: TaskSpec

hardware: HardwareProfile

program: Program

history: List[Action / Edit / Metrics]

Action / ProgramEdit

type of modification (insert/remove/replace/fuse/etc.)

target location (which gate, which qubits, which block)

parameters (angles, etc.)

These are the “semantic” objects. The LLM never sees them directly, only encodings/tokens.

3. Token Vocabulary Design

You’re not using English. You’re designing a program language for the LLM.

Split tokens into categories:

3.1. Program structure tokens

Basic structural markers:

PROGRAM_START, PROGRAM_END

BLOCK_START, BLOCK_END (for subcircuits)

LOOP_START, LOOP_END, LOOP_REPEAT_K

IF_START, IF_END, ELSE_START, ELSE_END (optional, higher‐level)

SUBROUTINE_DEF, SUBROUTINE_CALL

You might not implement all at v1, but the vocabulary is defined.

3.2. Gate tokens

A token per gate type, not per qubit:

Examples:

GATE_H (Hadamard)

GATE_X, GATE_Y, GATE_Z

GATE_RX, GATE_RY, GATE_RZ

GATE_CNOT, GATE_CZ, GATE_SWAP

GATE_TOFFOLI

GATE_U3, GATE_U1, GATE_U2 (if you use U-gates)

GATE_PHASE, GATE_T, GATE_S, etc.

GATE_CUSTOM_xxx for hardware-specific primitives

These are gate identities. Qubit indices and parameters are separate tokens.

3.3. Qubit / wire tokens

You don’t want infinite tokens, so you define a fixed set:

Q0, Q1, Q2, …, Q_N-1 (where N is max qubits your system supports)

optionally a Q_TARGET, Q_CONTROL, Q_ANCILLA abstraction if you make relative addressing

For multi-qubit gates, you emit tokens like:

GATE_CNOT Q1 Q3

GATE_CZ Q0 Q2

as a token sequence, not a single token.

So in token form:

GATE_CNOT  Q_CONTROL Q1  Q_TARGET Q3


or simpler:

GATE_CNOT  Q1  Q3


depending on how explicit you want.

3.4. Parameter tokens (angles, etc.)

Continuous values must be discretized.

Options:

Angle bins:
THETA_BIN_0, THETA_BIN_1, …, THETA_BIN_K
representing ranges like [0, 2π) split into K bins.

Parameter tokens for general reals:
e.g. PARAM_IDX_0, PARAM_IDX_1, … that index into a side-channel param array.

Simple practical scheme:

use ANGLE_xxx tokens for common values:

ANGLE_0, ANGLE_PI_2, ANGLE_PI_4, ANGLE_PI_8, ANGLE_PI, etc.

plus some binned tokens like ANGLE_BIN_0..ANGLE_BIN_15 for more fine-grained control

So for a rotation gate:

GATE_RZ  Q2  ANGLE_PI_4
GATE_RX  Q1  ANGLE_BIN_7

3.5. Action / edit tokens

You need to model actions in the search (not just static programs).

Define tokens like:

EDIT_INSERT

EDIT_DELETE

EDIT_REPLACE

EDIT_MOVE

EDIT_FUSE

EDIT_COMMUTE

EDIT_OPTIMIZE_BLOCK

EDIT_ADD_SUBROUTINE

EDIT_INLINE_SUBROUTINE

These will be combined with position tokens.

3.6. Position / location tokens

To specify where the edit applies:

POS_GATE_i (for gate index i in the sequence)

or: POS_LAYER_i, POS_BLOCK_j

or relative tokens: POS_BEFORE, POS_AFTER, POS_AROUND, POS_BLOCK_START, POS_BLOCK_END

Example action token sequence:

EDIT_REPLACE  POS_GATE_15  GATE_CNOT  Q1  Q2  ->  GATE_RZX  Q1  Q2  ANGLE_PI_4


In practice the model just predicts a token sequence that your parser interprets as a structured action.

3.7. Task tokens

To encode the TaskSpec:

TASK_TYPE_QFT, TASK_TYPE_GROVER, TASK_TYPE_PHASE_ESTIMATION, TASK_TYPE_QML, TASK_TYPE_RANDOM_UNITARY, etc.

TASK_NUM_QUBITS_k (k from 1..N)

TASK_OBJECTIVE_MIN_DEPTH

TASK_OBJECTIVE_MAX_FIDELITY

TASK_OBJECTIVE_BALANCED

TASK_RESOURCE_CONSTRAINT_xxx

These appear near the beginning of the sequence, so the LLM conditions on them.

3.8. Hardware tokens

To encode hardware:

HARDWARE_CONNECTIVITY_FULL

HARDWARE_CONNECTIVITY_LINEAR

HARDWARE_CONNECTIVITY_2DGRID

or explicit adjacency tokens like EDGE_Q0_Q1, EDGE_Q1_Q2 etc.

HARDWARE_GATESET_xxx (e.g. HARDWARE_GATESET_CZ, HARDWARE_GATESET_CNOT)

HARDWARE_NOISE_LOW, HARDWARE_NOISE_MEDIUM, HARDWARE_NOISE_HIGH

You don’t need insane detail in v1; coarse tokens are OK.

3.9. Special tokens

PAD, EOS, BOS, etc.

Possibly SEP to separate sections (task spec / hardware / program / action).

4. Model Architecture

The backbone is a standard decoder-only transformer (GPT-style), but:

Input tokens = your custom vocabulary

Output = program/action tokens

Additional heads = value and metrics prediction

4.1. Embeddings

You need:

TokenEmbedding[token_id] -> ℝ^d

PositionEmbedding[pos] -> ℝ^d (or RoPE)

Optionally separate embeddings for different semantic types:

GateEmbedding, QubitEmbedding, ActionEmbedding, etc., but usually one shared embedding + type embeddings is enough.

Final embedding for token i:

x_i = TokenEmbedding[token_i] + PositionEmbedding[i] + TypeEmbedding[type(token_i)]

4.2. Transformer layers

Standard stack of N layers:

Each layer l:

Multi-head self-attention:

Q = xW_Q, K = xW_K, V = xW_V

Attention = softmax(QKᵀ / sqrt(d)) V

causal mask: no attending to future tokens

Feedforward:

FFN(x) = W2 · activation(W1 · x)

Plus residual + layernorm.

You don’t need to reinvent this; you just describe:

“The Research LLM uses a decoder-only transformer with N layers, multi-head self-attention, and feedforward networks, trained under an autoregressive objective on program and action sequences.”

4.3. Heads

You have multiple heads coming out of the final hidden states:

4.3.1. Policy head

Input: hidden state at final position (or full sequence)

Output: probability distribution over next token or directly over actions.

Two approaches:

Token-level:

standard LM head: logits = hidden @ W_vocabᵀ

softmax → P(next_token | previous_tokens)

Action-level:

parse sequence into a “state embedding” (e.g., [last token, or special aggregated token])

then project to logits over discrete action space:
PolicyHead(state_embedding) -> logits_over_actions

For flexibility, you can keep token-level and parse structured actions externally.

4.3.2. Value head

Input: aggregated sequence embedding (e.g. mean pooling, special [CLS]-like token)

Output: scalar V ≈ expected future reward

V = wᵀ h_cls + b

Used for RL algorithms like PPO, A2C, etc.

4.3.3. Metrics head (optional)

Predicts:

fidelity_estimate

depth

2qubit_gate_count

noise_robustness

Super useful for pruning.

5. Environment + Search Loop

The LLM is used inside an environment that interacts with the simulator.

Define an environment:

QuantumAlgorithmDiscoveryEnv:
  - reset(task, hardware) -> state
  - step(action) -> (next_state, reward, done, info)
  - get_observation(state) -> token_sequence


state contains:

current Program

accumulated metrics

step counter, etc.

get_observation(state):

encodes:

task tokens

hardware tokens

encoded current program

maybe past actions summary

This observation sequence goes into the LLM.

The LLM outputs an action (by sampling or argmax over its policy).

Environment applies this action:

updates the Program

possibly runs a partial or full simulation

computes reward

returns new state

6. RL Training Loop (core of what you asked)

Here’s the conceptual PPO-like loop.

6.1. Step 0: Initialize

initialize LLM parameters θ

initialize value head parameters φ

set up environment + simulator + reward function

choose RL algorithm (PPO is a good mental model)

6.2. Step 1: Generate trajectories

For each iteration:

Sample a batch of tasks task ~ TaskDistribution

For each task:

state = env.reset(task, hardware)

for t in 0..T_max:

obs = env.get_observation(state)

tokenize: tokens = encode(obs)

run LLM forward:

get policy πθ(a|obs)

get value estimate Vφ(obs)

sample action a ~ πθ(a|obs) (or with some exploration strategy)

next_state, reward, done, info = env.step(a)

store transition:

(obs, tokens, action, reward, done, V_estimate)

state = next_state

if done: break

Collect many such trajectories into a replay buffer / batch.

Result: a dataset of (obs, action, reward, value_pred, done) sequences.

6.3. Step 2: Compute returns and advantages

For each trajectory:

Compute discounted returns:

G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...


Compute advantages (e.g., GAE):

A_t = G_t - Vφ(obs_t)


These are the learning signals.

6.4. Step 3: Policy update (e.g. PPO)

Using stored data with old policy πθ_old:

For each sample:

log prob of action under old policy: log πθ_old(a_t | obs_t) (stored)

recompute log prob under current θ: log πθ(a_t | obs_t)

ratio:

r_t = exp( log πθ(a_t|obs_t) - log πθ_old(a_t|obs_t) )


PPO objective:

L_policy(θ) = E_t[ min( r_t * A_t,
                        clip(r_t, 1 - ε, 1 + ε) * A_t ) ]


Value loss:

L_value(φ) = E_t[ (Vφ(obs_t) - G_t)² ]


Entropy bonus (for exploration):

L_entropy(θ) = E_t[ H(πθ(.|obs_t)) ]


Total loss:

L_total = -L_policy(θ) + c1 * L_value(φ) - c2 * L_entropy(θ)


Backprop → update θ and φ.

Do multiple gradient steps on this batch.

6.5. Step 4: Archive and dataset logging

Parallel to RL:

For each done episode, log:

task

final program

metrics

reward

full trajectory

Saved to DiscoveryArchive.

From time to time, you can:

build an offline supervised dataset:

(obs → good action) pairs from high-reward trajectories

pretrain / finetune the LLM with behaviour cloning on this dataset.

That’s how you get hybrid RL + supervised learning.

7. Runtime Functions / Interfaces (what your project exposes)

At the API level, you’ll want something like:

7.1. Policy usage
class ResearchLLMPolicy:
    def suggest_initial_program(self, task: TaskSpec, hardware: HardwareProfile) -> Program:
        ...

    def suggest_edit(self, state: ProgramState) -> Action:
        ...

    def score_program(self, task: TaskSpec, hardware: HardwareProfile, program: Program) -> float:
        ...  # approximate quality


Internally, each:

builds an observation / token sequence

runs through the transformer

decodes tokens into an action or program

7.2. High-level discovery loop
def discover_algorithms(task_set, hardware_set, budget) -> DiscoveryArchive:
    archive = DiscoveryArchive()
    for _ in range(budget.episodes):
        task = sample_task(task_set)
        hardware = sample_hardware(hardware_set)
        env = QuantumAlgorithmDiscoveryEnv(task, hardware)
        state = env.reset()

        for t in range(budget.max_steps_per_episode):
            action = research_llm_policy.suggest_edit(state)
            next_state, reward, done, info = env.step(action)
            archive.log_step(task, hardware, state.program, action, reward, info)
            state = next_state
            if done:
                archive.log_final(task, hardware, state.program, info.metrics)
                break
    return archive


RL training wraps around this (collect transitions, compute advantages, update model).

8. How to describe this in the README

You can structure the section like:

Research LLM Overview

transformer-based policy/value network for quantum program space

operates on program tokens, not natural language

Tokenization & Vocabulary

program structure tokens

gate tokens

qubit tokens

parameter tokens

action/edit tokens

task & hardware tokens

special control tokens

Model Architecture

embeddings (token, position, type)

decoder-only transformer

policy head, value head, optional metrics head

Environment & Observation

definition of QuantumAlgorithmDiscoveryEnv

observation construction from task, hardware, and current program

RL Training Loop

trajectory collection

returns/advantages

PPO-style update

archive logging and offline supervised fine-tuning

Runtime Interfaces

ResearchLLMPolicy functions

integration with discovery engine and simulator