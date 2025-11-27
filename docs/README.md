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
                                                |               ----------------------
                                                |         <-    |    OPTIMIZER NN    | 6.
                                                V               ----------------------
        ------------------           ------------------------
     5. |     QEC NN     |    ->     |       SIMULATOR      | 4.
        ------------------           ------------------------

## 1. ALGORITHM GENERATOR NEURAL NETWORK
- This is a Neural Network that is capable of developing new quantum algorithms, by generating "random" algorithms and circuits (not totally random, with the minimal logic to not create unusable algorithms and circuits) and grade them based on the perfomance predicted.

- ### COMPOSITION

    - #### 1. Data Representations
        - TaskSpec: Define the computational problem
        - HardwareProfile: 
            - [ ] topology
            - [ ] native gate set
            - [ ] error/noise characteristics
        - [ ] Program/Circuit
            - [ ] list of gates, qubits, parameters, structure
        - [ ] Action
            - [ ] type of modificatio (insert, remove, replace, fuse)
            - [ ] target location (which gate, which qubit, which block)
            - [ ] parameters (angles, etc)
        - [ ] ProgramState:
            - [ ] task: TaskSpec
            - [ ] hardware: HardwareProfile
            - [ ] program: Program
            - [ ] history: [Action / Edit / Metrics]
    
    - #### 2. Token Vocabulary
        ##### Program Structure Tokens
        - PROGRAM_START, PROGRAM_END
        - BLOCK_START, BLOCK_END
        - LOOP_STARTS, LOOP_END, LOOP_REPEAT
        - IF_START, IF_END, ELSE_START, ELSE_END
        - SUBROUTINE_DEF, SUBROUTINE_CALL

        #### Gate Type Tokens
        - [ ] GATE_H
        - [ ] GATE_X, GATE_Y, GATE_Z





## 2. DSL
- This is a very basic quantum language model, that is compiled by the quantum compiler. The objective for this language is to be powerfull enough to run the Algorithm Generator Neural Network.