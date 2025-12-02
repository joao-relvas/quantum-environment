# STRUCTURE

                                    ------------------------
    1.                              |     ALGO GEN NN      |
                                    ------------------------
                                                |
                                                V
                                    ------------------------
    2.                              |          DSL         |
                                    ------------------------
                                                |
                                                V
                                    ------------------------
    3.                              |        COMPILER      |
                                    ------------------------
        ------------------                      |
    4.  |  OPTIMIZER NN  |    ->                |
        ------------------                      |
                                                V
        ------------------           ------------------------
    5.  |     QEC NN     |    ->     |       SIMULATOR      | 6.
        ------------------           ------------------------

## 1. ALGORITHM GENERATOR NEURAL NETWORK
This is a Neural Network that is capable of developing new quantum algorithms, by generating "random" algorithms and circuits (not totally random, with the minimal logic to not create unusable algorithms and circuits) and grade them based on the perfomance predicted.

### COMPOSITION

#### 1. Data Representations
- TaskSpec: Define the computational problem
- HardwareProfile: 
    - topology
    - native gate set
    - error/noise characteristics
- Program/Circuit
    - list of gates, qubits, parameters, structure
- Action
    - type of modificatio (insert, remove, replace, fuse)
    - target location (which gate, which qubit, which block)
    - parameters (angles, etc)
- ProgramState:
    - task: TaskSpec
    - hardware: HardwareProfile
    - program: Program
    - history: [Action / Edit / Metrics]

#### 2. Token Vocabulary
##### Program Structure Tokens
- PROGRAM_START, PROGRAM_END
- BLOCK_START, BLOCK_END
- LOOP_STARTS, LOOP_END, LOOP_REPEAT
- IF_START, IF_END, ELSE_START, ELSE_END
- SUBROUTINE_DEF, SUBROUTINE_CALL

#### Gate Type Tokens
- GATE_H
- GATE_X, GATE_Y, GATE_Z
- GATE_RX, GATE_RY, GATE_RZ (rotation gate)
- GATE_CNOT, GATE_CZ, GATE_SWAP
- GATE_TOFFOLI
- GATE_U3, GATE_U1, GATE_U2
- GATE_PHASE, GATE_T, GATE_S

#### Qubit / Wire Tokens
- Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7
- Q_TARGET
- Q_CONTROL
- Q_ANCILLA
- Q_DATA_i
- Q_LOOP_INDEX
- Q_SELECTED
- Q_PAIR_NEXT
- Q_PAIR_PREV
- GATE_CNOT Q1 Q3

#### Parameter Tokens
- THETA_BIN_0, THETA_BIN_1, THETA_BIN_2 ... THETA_BIN_K (range [0, 2π] divided in K parts)
- ANGLE_0, ANGLE_PI_2, ANGLE_PI_4, ANGLE_PI_8, ANGLE_PI

#### Action Tokens
- EDIT_INSERT
- EDIT_DELETE
- EDIT_REPLACE
- EDIT_MOVE   
- EDIT_FUSE
- EDIT_COMMUTE    
- EDIT_OPTIMIZE_BLOCK
- EDIT_ADD_SUBROUTINE
- EDIT_INLINE_SUBROUTINE
(Combine with position tokens)

#### Position Tokens
- POS_BEFORE
- POS_AFTER
- POS_AROUND
- POS_BLOCK_START
- POS_BLOCK_END

#### Task Tokens
- TASK_TYPE_QFT, TASK_TYPE_GROVER, TASK_TYPE_PHASE_ESTIMATION, TASK_TYPE_QML, TASK_TYPE_RANDOM_UNITARY, etc.
- TASK_NUM_QUBITS_k (k from 1..N)
- TASK_OBJECTIVE_MIN_DEPTH
- TASK_OBJECTIVE_MAX_FIDELITY
- TASK_OBJECTIVE_BALANCED
- TASK_RESOURCE_CONSTRAINT_xxx
(To encode the TaskSpec)

#### Hardware Tokens
- HARDWARE_CONNECTIVITY_FULL
- HARDWARE_CONNECTIVITY_LINEAR
- HARDWARE_CONNECTIVITY_2DGRID
- HARDWARE_GATESET_xxx
- HARDWARE_NOISE_LOW, HARDWARE_NOISE_MEDIUM, HARDWARE_NOISE_HIGH

#### Special Tokens
- PAD, EOS, BOS ...

### 3. Model Architecture
#### Embedding
(Turn tokens into vectors for the NN to consume)

- TokenEmbedding[token_id] -> ℝ^d
- PositionEmbedding[pos] -> ℝ^d
- TypeEmbedding(token_i) -> ℝ^d

```x_i = TokenEmbedding[token_id] + PositionEmbedding[pos] + TypeEmbedding(token_i)```

#### Transformer Layers

Stack of N layers

Each layer I:
- Multi-head self-attention:
    ```Q = xW_Q, K = xW_K, V = xW_V```
    ```Attention = softmax(QKᵀ / sqrt(d)) V```
    ```causal mask: no attending to future tokens```

- Feedforward:
    ```FFN(x) = W2 · activation(W1 · x)```

- Residual + layernorm

#### Heads
- Policy Head
    ```Input: hidden state at final position.```
    ```Output: probability distribution over next token or directly over actions.```
    This has two aproaches:
    1. Token Level:
        Standard LM head: ```logits = hidden @ W_vocabᵀ```
        softmax → ```P(next_token | previous_tokens)```

    2. Action Level:
        Parse sequence into a "state embeding"
        Then project to logits over discrete action space:
        ```PolicyHead(state_embedding) -> logits_over_actions```

- Value Head
    Input: aggregated sequence embedding (e.g. mean pooling, special [CLS]-like token)
    Output: scalar V ≈ expected future reward
    ```V = wᵀ h_cls + b```

- Metrics Head
    ```fidelity_estimate```
    ```depth```
    ```2qubit_gate_count```
    ```noise_robustness```

### 4. Environment + Search Loop
- #### Quantum Algorithm Discovery Entry
    - reset(task, hardware) -> state
    - step(action) -> (next_state, reward, done, info)
    - get_observation(state) -> token_sequence

- State
    Contains:
    - Current program
    - Accumulated metrics
    - Step Counter
- get_observation(state)
    Encodes
    - task tokens
    - hardware tokens
    - encoded current program
    - maybe past actions summary

The LLM spits **an action** (argmax)

Environment applies this action:
- Update the program
- Run partial or full simulation
- Computes reward
- Returns new state

### 6. RL Training Loopx
- #### Step 0: Initialize
- Initialize LLM parameters θ
- Initialize value head parameters φ
- Set up environment + Simulator + Reward function
- Choose RL Algorithm (PPO is a good mental model)

- #### Step 1: Generate Trajectories
- ```1.``` Sample a batch of tasks task ~ TaskDistribution
- ```2.``` For each task:
    - obs = env.get_observation(state)
    - tokenize: tokens = encode(obs)
    - run LLM forward:
        - get policy πθ(a|obs)
        - get value estimate Vφ(obs)
    - sample action a ~ πθ(a|obs) (or with some exploration strategy)
    - next_state, reward, done, info = env.step(a)
    - store transition:
        - (obs, tokens, action, reward, done, V_estimate)
    - state = next_state
    - if done: break
- ```3.``` Collect many such trajectories into a replay buffer / batch.
```Result: a dataset of (obs, action, reward, value_pred, done) sequences.````

- #### Step 2: Compute returns and advantages
    For each trajectory:
    ##### Compute discounted returns:
    - ```G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...```

    ##### Compute advantages (e.g. GAE):
    - ```A_t = G_t - Vφ(obs_t)```

    These are learning signals.

- #### Step 3: Policy Update (e.g. PPO)
    Using stored data with old policy πθ_old:
    - ```1.``` For each sample:
        - log prob of action under old policy: log πθ_old(a_t | obs_t) (stored)
        - recompute log prob under current θ: log πθ(a_t | obs_t)
        - ratio:
            ```r_t = exp( log πθ(a_t|obs_t) - log πθ_old(a_t|obs_t) )```
    - ```2.``` PPO Objective:
        - ```L_policy(θ) = E_t[ min( r_t * A_t, clip(r_t, 1 - ε, 1 + ε) * A_t ) ]```
    - ```3.``` Value Loss:
        - ```L_value(φ) = E_t[ (Vφ(obs_t) - G_t)² ]```
    - ```4.``` Entropy bonus (for exploration):
        - ```L_entropy(θ) = E_t[ H(πθ(.|obs_t)) ]```
    - ```5.``` Total Loss:
        - ```L_total = -L_policy(θ) + c1 * L_value(φ) - c2 * L_entropy(θ)```
    - ```6.``` Backdrop -> Update θ and φ.

- #### Step 4: Archive and dataset logging
    Parallel to RL:
    For each done episode, log:
    - task
    - final program
    - metrics
    - reward
    - full trajectory
    Saved to DiscoveryArchive.

    From time to time, you can:
    - build an offline supervised dataset:
    - (obs → good action) pairs from high-reward trajectories
    - pretrain / finetune the LLM with behaviour cloning on this dataset.

    That’s how you get hybrid RL + supervised learning.

### 7. Runtime Functions / Interfaces
- #### Policy Usage
    I need something like:
    ```class ResearchLLMPolicy:
        def suggest_initial_program(self, task: TaskSpec, hardware: HardwareProfile) -> Program:
        ...

        def suggest_edit(self, state: ProgramState) -> Action:
        ...

        def score_program(self, task: TaskSpec, hardware: HardwareProfile, program: Program) -> float:
        ...  # approximate quality
    ```

    Each one:
    - builds an observation/token sequence
    - runs through the transformer
    - decodes tokens into an action or program

- #### High-level Discovery Loop
    Should look something like this
    ```def discover_algorithms(task_set, hardware_set, budget) -> DiscoveryArchive:
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
    ```
RL training wraps around this (collect transitions, compute advantages, update model).

### Resume
1. ```Research LLM Overview```
    - transformer-based policy/value network for quantum program space
    - operates on program tokens, not natural language
2. ```Tokenization & Vocabulary```
    - program structure tokens
    - gate tokens
    - qubit tokens
    - parameter tokens
    - action/edit tokens
    - task & hardware tokens
    - special control tokens
3. ```Model Architecture```
    - embeddings (token, position, type)
    - decoder-only transformer
    - policy head, value head, optional metrics head
4. ```Environment & Observation```
    - definition of QuantumAlgorithmDiscoveryEnv
    - observation construction from task, hardware, and current program
5. ```RL Training Loop```
    - trajectory collection
    - returns/advantages
    - PPO-style update
    - archive logging and offline supervised fine-tuning
6. ```Runtime Interfaces```
    - ResearchLLMPolicy functions
    - integration with discovery engine and simulator
        
## 2. DSL
This is a very basic quantum language model, that is compiled by the quantum compiler. The objective for this language is to be powerfull enough to run the Algorithm Generator Neural Network.

### Composition
#### 1. DSL Frontend
- Clean surface to let users or AI to write quantum programs. Converts raw DSL code to tokens for the parser.
What is needed:
- A class or module responsible for **parsing the DSL input text.**
- It must understand:
    - keywords (gate names, control structures, declarations)
    - punctuation (parentheses, commas, semicolons)
    - parameters (angles, numeric values)
    - identifiers (variable names, qubit names)
- Tokenization rules:
    For example
    - any sequence of letters -> identifier token
    - something like RX(π/2)  -> gate with param

#### 2. DSL Parser
- Takes tokens and transform into a strutured AST (Abstract Syntax Tree).
What is needed:
- A parser class that:
    - accepts the token stream
    - enforces grammar rules
    - throws meaningful syntax errors
- It should produce a tree-like structure:
    - program
    - declarations
    - blocks
    - gate instructions
    - control instructions
    - loops
    - conditionals
    - subroutine definitions
- **A gramar specification** (BNF-like) describing:
    - how circuits are declared
    - how qubits are bound
    - how gates are applied
    - how loops and quantum blocks work

#### 3. Abstracty Syntax Tree (AST)
Represents a program in a language-agnostic way.
What is needed:
- A set of node types for:
    - entire program
    - qubit declarations
    - classical vars
    - gate applications
    - controlled gates
    - measurement statements
    - loops (for, while)
    - conditionals (if/else)
    - subroutine definitions
    - subroutine calls
    - parallel blocks (optional)
- A base "Node" class and child classes for each construct

#### 4. Type Checker / Semantic Analyzer
Validates the meaning of the AST
What is needed:
- A class/module that:
    - ensures gates reference valid qubits
    - checks parameter types (angles are real numbers, loops are integers)
    - verifies hardware compatibility (e.g., multi-qubit gates only on connected qubits)
    - ensures subroutine usage is valid (arity, parameter - )
    *Optional:*
    - Scope checking for variables or let-bindings
    - Checking that classical/quantum operations aren’t mixed incorrectly

#### 5. DSL -> IR Translator (Lowering Pipeline)
Lowering mechanism that converts the AST into my intermediate representation.
What is needed:
- A translator class that:
    - takes an AST as input
    - walks through each node
    - converts it into the IR instructions:
        - OpGate(...)
        - OpMeasure(...)
        - OpControl(...)
        - OpBlock(...)
- It should expand high-level constructs like:
    - loops into repeated blocks of IR
    - conditionals into IR control-flow primitives (if you support them)
    - subroutines into callable IR fragments
DSL -> AST -> IR

#### 6. IR Optimizer Hooks
Basic IR optimizations before sending to the compiler.
What is needed:
- A "PassManager" class that can run optimization passes such as:
    - gate fusion
    - cancellation (H then H cancels)
    - commuting gates for better structure
    - removing no-op blocks
    - inline/specialize subroutines
- The DSL doesn't implement these, it only prepares the IR. The compiler handles deeper optimizations.

#### 7. Error Handling and Diagnostics
Error messages
A layer that:
- catches parse errors with line/column info
- catches semantic errors with suggestions
- catches invalid control-flow or syntax misuse
- integrates with VS Code extensions (optional)
- Examples:
- “Unknown gate HHHH on line 12”
- “Qubit q7 not declared before use”
- “CNOT requires 2 qubits but got 1”

#### 8. DSL Runtime (Optional)
May want a thin runtime to:
- bind parameterized values
- evaluate classical expressions
- support symbolic parameters
- precompute angles or reduce expressions like pi/4 - pi/8

#### 9. DSL Library / Built-in functions
The DSL should have a library of built-in quantum or not functions to simplify the coding.
Examples:
- HADAMARD_LAYER(qubits[])
- ENTANGLE_RING(qubits[])
- QFT(n)
- GROVER_DIFFUSER()
- AMPLITUDE_ENCODE(data[])

#### 10. DSL Grammar   
- Complete grammar document describing:
- keywords
- syntax
- operators
- gate application format
- control structure format
- allowed expressions
- naming rules

Example:
``` program        ::= stmt_list
    stmt_list      ::= stmt (";" stmt)*
    stmt           ::= gate_call
                   | if_stmt
                   | loop_stmt
                   | subroutine_def
    gate_call      ::= IDENTIFIER "(" arg_list ")" IDENTIFIER
    arg_list       ::= expr ("," expr)*
```

#### 11. DSL-to-Token Encoder (Tokenizer for the Research LLM)
Transforms a version of the DSL into special program tokens for the LLM
It needs:  
- A class that walks the IR and produces token sequences like:
- TASK_*
- HARDWARE_*
- GATE_*
- Q*
- ANGLE_*
- EDIT_* (for RL mode)
- SUB_START, LOOP_START, etc.
The LLM learns from these tokens, executes RL over them, and generates new tokens that correspond to valid DSL/IR constructs.

#### 12. DSL Interpreter for LLM Outputs
An Interpreter Class for when the LLM spits token sequences describing:
- a program
- an edit
- a subroutine
- a block
The class will then parse the sequence back into:
- AST, or
- IR Program, or
- program edit tuple

#### 13. DSL Validation Layer
Validates the LLM output.
The validator should:
-  checks if the generated program/edit is syntactically valid
-  checks semantic validity (correct qubits, valid parameters)
-  optionally auto-repairs simple mistakes (missing parentheses, wrong commas)
-  rejects hopelessly invalid outputs early

#### 14. DSL "Pretty Printer"
For debugging and documentation, it is needed a class that converts IR or AST into nice-looking DSL code.
Lets the LLM to expose its new algorithm discovers for the user.

#### 15. DSL Documentation layer
Must include:
- a formal description
- examples
- syntax reference
- list of built-in gates
- list of control structures
- how to define subroutines


## 3. Compiler
### Composition

#### 1. Compiler Frontend
First "real" compiler layer.

**This is a module that receives the AST from the DSL parser**

It understands:
- the structure of the AST
- declarations of qubits
- gate calls
- loops, blocks, subroutines
- parameters (angles, reals, symbolic expressions)

Then, the module:
- take the AST and normalize it
- remove syntatic sugar
- expand short-hand constructs
- validate the program structure
- verify that it fits the grammar and semantic rules of your language beyond what the DSL already checked

#### 2. Semantic / Type Analysis Layer
The compiler must validate deeper compiler-level semantics.

This has a Semantic Analyzer module that:
- checks qubit usage
- ensures gate arity is correct
- validates multi-qubit operations
- confirms gate exist on the target hardware
- ensures all references are in scope
- rejects impossible control structures (like classical branching dependent on quantum values unless supported)

And:
- annotate the AST with:
    - qubit resource usage 
    - parameter domains
    - known constants
    - hardware compatibility tags

#### 3. The IR Generator
Walks the AST and emits the Quantum IR

This part:
- convert each AST node into IR nodes
- flatten blocks into IR sequences
- unroll loops 
- inline or reference subroutines
- generate control-flow IR for if/else (if supported) - it will be supported
- resolve parameter expressions
- annotate IR with qubit indices and parameter values

#### 4. The IR Optimizer
Here is where the compiler turns intelligent.

It has a Pass Manager that can run a sequence of optimization passes

The pass should:
- **Low-level gate Optimizations**
    - cancel pairs of gates:
        - X followed by X
        - H H
        - Rz(a) Rz(b) -> Rz(a + b)
    - fuse contiguous rotation gates
    - collapse indentity operations
- **Commutation-based optimizations**
    - reorder gates that commute
    - push costly multi-qubit gates outward or inward strategically
    - separate or group gates for better structure 

- **Hardware-aware passes**
    - reroute multi-qubit gates according to connectivity
    - insert SWAPs only when necessary
    - choose best native decomposition depending on hardware profile

- **Higher-level passes**
    - detect repeated blocks → propose subroutines
    - detect known patterns (QFT, Grover diffuser, entangler blocks)
    - rewrite circuits into canonical forms

- **Structural optimizations**
    - flatten nested blocks
    - inline small subcircuits
    - specialize parameterized substructures

#### 5. Hardware-Abstraction Layer (HAL)
This module:

- Understands the target hardware:
    - connectivity graph
    - native gate set
    - allowed two-qubit operations
    - error rates
    - hardware-specific decomposition rules

- And applies:
    - gate decomposition (e.g., Toffoli → CZ+H+T sequence)
    - routing algorithms (like SABRE, token swapping, or your own)
    - noise-aware transformations (optional)

#### 6. The IR → Lower-Level Hardware IR Conversion
Some compilers use multiple IR layers:
- High-level IR (with subroutines, abstract gates)
- Mid-level IR (hardware-aware but still abstract)
- Low-level IR (pure native gates)

I should choose if it will have multiple layers or not.

But it will be needed a Lowering Pipeline that:
- expands all high-level constructs
- resolves all hardware-dependent decisions
- outputs the final executable circuit

#### 7. Validation & Verification Passes
Validate LLM.

I need passes for:
- **Logical validation**
    - qubit count integrity
    - no invalid wires
    - no dangling gate parameters

- **Circuit well-formedness**
    - consistent time-ordering of gates
    - no illegal control flows (if quantum control is unsupported)
    - ensure atomic operations are valid

- **Hardware validity**
    - all gates must be invocable on the target device
    - no illegal multi-qubit operations

#### 8. Compiler Stats & Metadata Extraction
This is essential for:
- debugging
- compiler feedback loops
- LLM-conditioned optimization
- simulation metrics

This module should compute:
- **Program metrics**
    - depth
    - width (qubit count)
    - t-gate count
    - Clifford count
    - two-qubit gate count
    - decomposition statistics

- **Hardware metrics**
    - routing overhead
    - fidelity estimates
    - noise susceptibility

#### 9. IR → Token Encoder (For LLM)
This bridges the compiler to the Research-LLM.
This module takes the IR and emits:
- ```GATE_*``` tokens
- ```Q*``` tokens
- ```ANGLE_*``` tokens
- structural tokens (```BLOCK_START```, ```LOOP_START```, etc.)
- hardware tokens
- task tokens

#### 10. Pretty-Printing / Inverse Lowering
This module:
- takes IR
- reconstructs a readable DSL program
- optionally reconstructs high-level ideas (visualization)

#### 11. Compiler Service API (External Interface)
The compiler is a layer, not a binary.

This should expose functions like:

- **High-level interfaces**
    - compile_program_from_dsl(source_code, hardware_profile)
    - compile_from_ast(ast, hardware_profile)

- **Internal interfaces**
    - lower_ir(ast)
    - optimize_ir(ir, passes)
    - apply_hardware_lowering(ir, hardware)

- **Export interfaces**
    - export_ir()
    - export_token_sequence()
    - export_compiler_stats()

#### 12. Integration with Research LLM
The compiler isn't just a compiler — it's part of the algorithm discovery loop.

This is a class that:
- receives LLM-generated actions/edits
- applies them to IR
- revalidates IR
- re-optimizes
- sends metrics to the simulator
- forwards results back to the RL learner

This allows:
- search-based exploration
- RL-guided optimization
- iterative circuit improvement

#### 13. Compiler Diagnostics & Tracing
Optional tools to:
- record each optimization pass
- show before/after circuits
- track transformation sequences
- provide debugging info for research

#### 14. Formal Compiler Documentation
The compiler needs reference documentation:
- IR specification
- optimization pass descriptions
- hardware lowering rules
- naming conventions
- example transformation pipelines

