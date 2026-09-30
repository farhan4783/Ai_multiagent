# Multi-Agent DRL System Implementation Plan

## Executive Summary  
Multi-agent deep reinforcement learning (MARL) extends single-agent RL to settings with multiple interacting agents (e.g. teams of robots, autonomous vehicles, game players).  MARL introduces challenges – non-stationarity (agents’ policies change during training), partial observability and credit assignment issues – and can be cooperative, competitive or mixed.  This project will build a full MADRL framework with clear goals, modular architecture, and robust training/evaluation pipelines.  The architecture will support *centralised training with decentralised execution* (CTDE), allowing joint learning (e.g. via central critics) while agents act independently at run-time. We will implement key algorithms (value-based and policy-gradient) and multi-agent variants (e.g. MADDPG, QMIX, COMA, MAPPO/RMAPPO), deploy in various environments (OpenAI Gym, PettingZoo, Unity ML-Agents, Isaac Gym) and use distributed training tools (e.g. Ray RLlib, IMPALA).  Rigorous evaluation (benchmarks, metrics, ablations) and monitoring (TensorBoard/WandB logs, reproducibility checks) will be included.  A Docker/Kubernetes-based CI/CD pipeline will ensure code quality and reproducibility.  Table 1 compares representative MARL algorithms and Table 2 compares simulation platforms. Detailed risk mitigation and ethical reviews (safety, fairness) are included.  

## Goals and Scope  
The primary goal is to develop a reusable **MADRL framework** that can train and deploy multi-agent policies in complex environments (e.g. robot teams, traffic simulations, multi-player games). Scope includes: defining project milestones, designing system architecture (agents, environment, trainer), selecting algorithms, building training infrastructure, and establishing evaluation pipelines.  The framework should handle both *cooperative* and *competitive* settings, with flexibility for heterogeneous agents or shared policy architectures.  Crucially, it will address MARL-specific challenges: non-stationarity, partial observability, multi-agent credit assignment and scalability. By leveraging *centralised training with decentralised execution* (CTDE), agents can learn from global state/action information but operate solely on local observations at run-time.  

Applications span autonomous vehicles, robotics, UAV fleets, networked games and simulations.  Success will be measured by training stable multi-agent policies that achieve specified team objectives under realistic constraints.  Ethical/legal considerations include ensuring safe agent behaviors (avoiding catastrophic or biased actions) and compliance with data/privacy regulations if any real data are used.  

## System Architecture  

**Components & Data Flows:** The system comprises (a) **Environment** simulators, (b) **Agent modules** (policy networks, communicators), (c) **Experience buffers**, (d) **Learner(s)** (central trainer), and (e) **Logging/monitoring**. At each step, the environment emits observations per agent; agents select actions (possibly communicating through a defined protocol); the environment applies joint action, returns next states and rewards; transitions are stored in replay buffer(s). Periodically, the central learner samples from buffers to update network weights. The **data flow** is cyclic: Environment → Agents → Actions → Environment → Rewards/Next-States → Buffer → Learner → Updated Policies.  

**Agent Types & Communication:** Agents may be *homogeneous* (sharing network weights) or *heterogeneous*. Communication can be implicit (shared parameters) or explicit: e.g. message-passing channels where agents exchange discrete/continuous signals. We will design an API for optional inter-agent messages. In CTDE mode, training can use global state or other agents’ actions, but agents will only access local observations at execution.  

**Centralised vs Decentralised Execution:** We adopt CTDE. During training, a **central critic** (or joint-value network) can condition on all agents’ states/actions, enabling more informed updates. At inference time, only each agent’s actor (policy) is used, acting independently. This matches algorithms like MADDPG and COMA. We will also support *fully decentralized training* scenarios (e.g. independent Q-learning) if needed.  

```mermaid
flowchart LR
    subgraph Environment
      E(Environment Simulator)
    end
    subgraph Agents
      A1(Agent 1 Policy) -->|Action a₁| E
      A2(Agent 2 Policy) -->|Action a₂| E
      A3(Agent 3 Policy) -->|Action a₃| E
    end
    E -->|obs₁,r₁| A1
    E -->|obs₂,r₂| A2
    E -->|obs₃,r₃| A3
    A1 & A2 & A3 -->|experiences (s,a,r,s')| Buffer
    Buffer -->|batch of joint experiences| Learner
    Learner -. updates .-> A1 & A2 & A3
    Learner -. updates .-> Buffer
    Learner -. logs .-> Monitoring[Monitoring / TensorBoard]
```
*Figure 1: System architecture. Multiple agents interact with a shared environment and send experience to a central learner. The learner updates agent policies and logs training progress.*  

**Communication Protocols:** If enabled, agents may send explicit messages each timestep. For example, we could support fixed-size vector messages or discrete signals. The architecture will include an optional “communication channel” integrated into agent observations. During training, communication can be differentiable and learned (akin to methods in learned communication MARL). We will design an interface for agent<–>agent messaging and allow both fixed pre-defined protocols and learned protocols.

## Algorithms Considered  

We will implement and compare a range of RL algorithms, categorised by their paradigms (value-based, policy-gradient, hybrid) and MARL extensions. Key candidates include:

- **Value-Based Methods:**  
  - *Independent Q-Learning* (IQL): Each agent learns its own Q-network via DQN-like updates, ignoring non-stationarity (baseline).  
  - *DQN (Deep Q-Network)*: Single-agent value-based method; forms basis for IQL. It uses a CNN with Q-learning to map raw pixels to Q-values.  
  - *Value-Decomposition Network (VDN)*: A cooperative MARL approach where a team-value Q-function is learned as the sum of per-agent Q-values, enabling decentralised execution while optimising a global return.  
  - *QMIX:* Extends VDN by using a learnable mixing network (with a monotonicity constraint) to combine per-agent Qs into a joint Q. QMIX can represent more complex value decompositions than VDN.  
  - *Tree-based Value Methods:* (Optional) e.g. deterministic policy gradient adapted to discrete actions.  

- **Policy-Gradient and Actor-Critic:**  
  - *REINFORCE/PG:* Basic policy-gradient. Known to suffer high variance as agents increase (cited below).  
  - *PPO (Proximal Policy Optimization)*: A stable policy-gradient method; easily extended to multi-agent by giving each agent its own PPO learner. PPO alternates between sampling and optimizing a surrogate clipped objective for multiple mini-batch epochs.  
  - *MADDPG (Multi-Agent DDPG)*: A popular CTDE algorithm using decentralized actor networks and a centralized critic for each agent. It enables continuous actions in coop/comp scenarios by extending DDPG to multiple agents.  
  - *COMA (Counterfactual MA PG)*: A fully cooperative actor-critic method. Actors (policies) are decentralized; a centralised critic estimates joint Q. It uses a counterfactual baseline to assign credit to each agent (comparing actual joint action to counterfactual where one agent’s action is marginalised out).  
  - *MAPPO (Multi-Agent PPO):* A variant applying PPO in CTDE mode; effectively multiple PPO agents with a shared or central critic. (For example, the OpenAI cPET example uses PPO with a centralized value function for MARL.)  
  - *R-MAPPO (Robust MAPPO):* A proposed extension focusing on robustness (e.g. by adding adversarial critics or noise). This is cutting-edge; if time permits, an R-MAPPO variant will be prototyped.  

- **Other Architectures:**  
  - *Actor-Critic (A2C/A3C):* Synchronous (A2C) or asynchronous (A3C) multi-agent training. Not CTDE by default, but can be adapted.  
  - *Value Decomposition Methods (QC, QTRAN):* We may mention methods like QTRAN (which relaxes VDN constraints) and QMIX’s variants.  
  - *Communication-Enabled Methods:* E.g. CommNet, TarMAC etc., where agents learn to share messages as actions. If explored, these would be part of “communication protocols” experiments.  

The following table summarises key algorithms:

| **Algorithm**            | **Type**           | **Training**         | **Use-case**                   | **Key Reference**           |
|--------------------------|--------------------|----------------------|--------------------------------|-----------------------------|
| **DQN (IQL)**            | Value-based        | Decentralised        | Discrete actions, baseline     | Mnih *et al.*  |
| **VDN**                  | Value decomposition| CTDE (sum decomposition) | Cooperative tasks            | Sunehag *et al.* (2017) |
| **QMIX**                 | Value mixing       | CTDE (monotonic)     | Cooperative, complex mixing    | Rashid *et al.* (2018) (not cited) |
| **MADDPG**               | Actor-Critic      | CTDE (central critic)| Mixed coop/comp, continuous    | Lowe *et al.* |
| **COMA**                 | Actor-Critic      | CTDE (central critic)| Fully cooperative, credit assignment | Foerster *et al.* |
| **PPO**                  | Policy gradient   | Decentralised (shared) | Any standard env            | Schulman *et al.*  |
| **MAPPO**                | Actor-Critic      | CTDE (central critic)| Multi-agent variant of PPO     | (OpenAI blog)                |
| **R-MAPPO**              | Actor-Critic      | CTDE (robustified)   | Robust variants of MAPPO       | (emerging research)          |

*Table 1: Comparison of selected MARL algorithms. “CTDE” = Centralised Training, Decentralised Execution. (All MARL methods require experience replay or parallel sampling; see text.)*  

Key observations (citing [27]): *single-agent methods struggle in MARL*: DQN’s reliance on replay breaks in non-stationary environments, and vanilla policy gradients suffer high variance with many agents.  Thus algorithms using centralized critics (e.g. MADDPG, COMA) or value decomposition (VDN/QMIX) are preferred.  

## Environment Design and Simulators  

Choosing or building environments is critical. We will leverage existing libraries and simulators:

- **OpenAI Gym/Gymnasium**: A standard RL interface for single-agent tasks (CartPole, Atari, MuJoCo, etc.). Gymnasium (maintained Gym fork) provides a wide suite of tasks (control, robotics, Atari) with a simple API. For MARL, Gym can be used by wrapping multi-agent environments into Gym (e.g. vectorized or multi-agent wrappers).
- **PettingZoo**: A dedicated MARL library with a unified API. It includes *over 50* built-in multi-agent environments (e.g. cooperative “pistonball”, competitive “knights_archers_zombies”). PettingZoo supports both *sequential* and *parallel* action models and can wrap environments in Gym-like interfaces. We will use PettingZoo for training standard MARL benchmarks and for prototyping new tasks.  
- **Unity ML-Agents**: An open-source toolkit to use Unity 3D/2D simulators for RL. ML-Agents allows designing complex multi-agent game-like environments (e.g. team sports, vehicle navigation). It includes example multi-agent scenarios and supports both cooperative and competitive training. Unity environments can be controlled via Python and also wrap to Gym/PettingZoo interfaces. We will use Unity ML-Agents for 3D tasks (e.g. multi-robot navigation, crowd simulations).  
- **NVIDIA Isaac Gym/Sim**: GPU-accelerated physics simulator for robotics (now part of Isaac Sim). Although recently deprecated, it allowed high-speed simulation of many robots on a GPU. If available, it could accelerate large-scale MARL in robotics. Otherwise, we may use MuJoCo or Bullet for multi-robot tasks.  
- **Other MARL Benchmarks**:  
  - *StarCraft Multi-Agent Challenge (SMAC)*: Standard cooperative-combat tasks.  
  - *Multi-Agent Particle Environments (MPE/MPE2)*: Simple 2D physics tasks (predator-prey, cooperative navigation).  
  - *Hanabi*: Fully cooperative imperfect-information card game (tests credit assignment).  
  - *Pommerman*: Competitive grid-world (pizza bomber game) for 4 agents.  
  - *Custom Environments*: We will implement domain-specific environments as needed, using Gym/PettingZoo APIs.  

Table 2 summarizes these options:

| **Simulator / Env.**         | **Type**         | **Agents**    | **Remarks**                                                             |
|-----------------------------|------------------|---------------|-------------------------------------------------------------------------|
| **Gymnasium (Gym)**    | 2D/3D, single-agent base | Single/Multi (via wrappers) | Standard RL API, many classic tasks (Atari, Mujoco, control).           |
| **PettingZoo**        | 2D, multi-agent  | Multiple      | Standard MARL API, many built-in cooperative/competitive environments. |
| **Unity ML-Agents**    | 3D/2D, game-like | Multi-agent   | Rich 3D environments; supports coop and adversarial tasks; Python API. |
| **Isaac Gym/Sim**           | 3D, robotics     | Multi-robot   | GPU-accelerated physics simulation (legacy); scales to thousands of bots. |
| **SMAC (StarCraft)**        | 3D, RTS game     | Multi-agent   | Cooperative combat mini-games (common MARL benchmark).                  |
| **Custom Env. (Gym/PZ)**    | Various          | As needed     | Domain-specific (e.g. traffic, navigation); implement with Gym/PD API.  |

*Table 2: Simulation platforms for MARL.*  

Each environment choice entails different compute requirements (Unity and Isaac are more GPU/physics intensive) and observation types (pixels vs vectors). We will select based on project needs: e.g. fast Gym/PettingZoo tasks for prototyping, Unity or Isaac for high-fidelity scenarios.  

## Training Pipeline and Infrastructure  

A robust training pipeline will automate data collection, model updates, and monitoring. Key elements:

- **Data Collection:** Rollout workers (environments) generate experience in parallel. We will use vectorized environments (e.g. PettingZoo’s async or parallel wrappers, Unity’s multiple Unity instances) to feed data to learners. Following distributed architectures (e.g. IMPALA), actors will send full trajectories or mini-batches to a central learner. Using IMPALA-style design, actors *decouple* from learning and stream experience; a GPU-based learner then performs batched gradient updates. This provides high throughput and stability.  
- **Replay Buffer:** For off-policy methods we will use experience replay. Options: (a) **Per-agent buffers** (each agent has own buffer), or (b) **Global buffer** storing joint transitions. Prioritised Experience Replay (PER) can be implemented if beneficial. However, as noted by Lowe *et al.*, non-stationarity can make naive replay unstable; CTDE algorithms mitigate this by using centralized critics or on-policy updates. We will initially use standard replay and later explore PER and other fixes.  
- **Algorithm Implementation:** We will build on established RL libraries: **Ray RLlib** (provides multi-agent support and distributed algorithms), **Stable Baselines3** (PPO, DQN single-agent), **Tianshou/CleanRL** for flexible pipelines. PettingZoo’s AgileRL and cleanrl tutorials can accelerate development. Critical components (e.g. MADDPG, QMIX, MAPPO) will be implemented or integrated from community implementations, ensuring consistency.  
- **Hyperparameter Tuning:** Use tools like **Ray Tune** or **Optuna** for automated HP search. Given the complexity of MARL, we will tune learning rates, discount factors, exploration noise, network sizes, etc. Tuning can run in parallel on clusters to find robust configurations.  
- **Distributed Infrastructure:**  
  - We anticipate training on GPU clusters. For moderate experiments, a single multi-GPU workstation may suffice; for larger scale we will use cloud (AWS/GCP with GPU instances) or on-prem HPC. Ray RLlib natively distributes workers across nodes.  
  - For example, an IMPALA agent can run thousands of actors on CPU while learner on GPU. We will containerize the code (Docker) for easy deployment on Kubernetes or Slurm clusters. A Kubernetes-based pipeline (Jenkins/Argo) will manage experiments, enabling reproducibility (see CI/CD below).  
- **Checkpoints and Reproducibility:** Training code will regularly save model checkpoints and RNG states. We will log hyperparameters and seeds. Using MLflow or Weights & Biases, each experiment’s config will be tracked. Code and data will be version-controlled (Git).  Seeds will be fixed (across envs, network inits) for reproducible runs.  
- **Logging & Monitoring:** Training progress (losses, rewards, metrics) will be logged. We will integrate **TensorBoard** (for simple scalar/graph logging) and **Weights & Biases** (for collaborative logging, hyperparam sweeps). Custom dashboards may display multi-agent-specific metrics (e.g. team reward vs individual reward). System metrics (GPU/CPU utilization) will be monitored to detect bottlenecks.  
- **Hardware Requirements:** A sample baseline compute setup: 4–8 NVIDIA GPUs (e.g. A100), multi-core CPUs, 128GB RAM. Unity/Isaac training may require more GPU memory for rendering. If budget allows, TPUs or additional GPUs can be used for large-scale runs.  We will profile and scale as needed.  

## Evaluation Metrics, Benchmarks, and Testing  

Rigorous evaluation is essential. We will measure both performance and robustness:

- **Reward and Success Metrics:** The primary metric is cumulative reward (or team reward) in the environment. For competitive tasks, win-rate against fixed opponents; for cooperative, task completion or team score. We may also track fairness (e.g. variance of rewards among agents) or efficiency (time to completion).  
- **Standard Benchmarks:** Use established MARL benchmarks, such as: 
  - **StarCraft II (SMAC)** – cooperative micromanagement tasks.  
  - **PettingZoo challenges** (e.g. KAZ, cooperative pong, etc.).  
  - **Multi-Agent MuJoCo** – tasks like cooperative locomotion.  
  - **Hanabi** – tests communication and credit assignment (agents share full or partial observation).  
  - **Resource-sharing or navigation tasks** in Unity (e.g. cooperative robots).  
  - We will compare against published results for these tasks.  
- **Ablation Studies:** To understand design choices, we will systematically remove or vary components: e.g. disable communication channel, use only local critics vs centralized critic, vary number of agents, or remove reward shaping. Such ablations will quantify the effect of each module.  
- **Robustness and Safety Testing:** Agents will be tested under variations: noisy observations, random agent failures, or adversarial opponents. We will check that learned policies degrade gracefully. Safety tests may include constraints (e.g. no collisions). If applicable, formal safe-RL constraints (e.g. Shielded RL) may be evaluated.  
- **Reproducibility:** We will fix random seeds across multiple runs and report mean±std performance to ensure stability. All hyperparameters and code versions will be documented to allow independent reproduction of results.  
- **Benchmarks and Reporting:** We will report results in tables/plots, comparing algorithms on each task. Performance (final reward), sample efficiency (learning curve), and robustness metrics will be included. 

## Dataset and Replay Strategies  

Although RL typically generates its own data, some considerations arise in MARL:

- **Replay Buffers:** For off-policy methods (DQN variants, MADDPG), each agent may have its own replay buffer of (obs,action,reward,next_obs) tuples. Alternatively, a **joint buffer** can store full joint states and actions. CTDE methods like COMA/MADDPG often use joint samples. We will experiment with both designs.  
- **Prioritisation:** **Prioritised Experience Replay** (Schaul *et al.*) can be applied per-agent or globally to focus on high-TD-error transitions. Care must be taken: Lowe *et al.* note that naive replay is destabilised by non-stationarity, but central critics mitigate this. We will start with uniform replay and add PER if needed.  
- **Data Efficiency:** Model-based or imitation learning is beyond scope, but for data efficiency we can incorporate **curriculum learning** (start with simpler tasks, increase difficulty) and **data augmentation** (domain randomisation in Unity).  
- **External Datasets:** If any offline expert trajectories are available (e.g. human plays, logged data), we can mix them via imitation or off-policy RL. However, primary training is online in simulation.  

## Code Structure, CI/CD, Testing, Deployment  

**Code Organization:** We will use Python with modular design. For example:  
- `env/`: environment wrappers (Gym, PettingZoo, Unity connectors).  
- `agents/`: policy and network definitions, possibly one class per algorithm.  
- `train/`: training loops and pipelines.  
- `utils/`: common utilities (replay buffer, logging, hyperparam parsing).  
- `config/`: YAML/JSON config files for experiments.  
- `tests/`: unit tests for key components.  

Code will adhere to style guides (e.g. PEP8) and use type hints. We will document interfaces so that new algorithms or environments can be plugged in easily.  

**CI/CD:** We will set up continuous integration (e.g. GitHub Actions or GitLab CI) to automatically run unit tests on commits. Test coverage will include: sanity checks that environments reset correctly, agents produce valid actions, and that basic training loops can run without errors. We may include minimal “smoke tests” of training on a trivial env. Static code analysis (flake8, mypy) will enforce code quality.  

**Testing:** Beyond CI, integration tests will run a few training steps in a simple environment (e.g. a 2-agent GridWorld) to ensure full-pipeline functionality. We will validate outputs (e.g. loss decreases) to catch runtime issues.  

**Deployment:** For deployment (e.g. sharing or production), we will use **Docker** to containerize the code with specific dependencies. For example, an image may include Python, PyTorch, Gym, etc. Deployment on Kubernetes (or cloud ML services) will allow scaling experiments. We will also prepare reproducible experiment scripts, so that retraining or demos can be launched with a single command.  

**Monitoring & Logging:** In production-like settings, we will integrate monitoring of resource usage (Prometheus/Grafana) and logs of policy actions. If deploying agents (e.g. in simulation), we can use game engine diagnostics or custom monitors.  

## Timeline and Milestones  

A proposed timeline (subject to adjustment):

- **Month 0–2:** *Setup & Prototyping* – Define detailed requirements, set up repositories and CI. Implement basic Gym/PettingZoo env and a simple MARL algorithm (e.g. independent DQN) to test pipeline. Select initial hardware (GPU cluster or cloud).  
- **Month 3–5:** *Algorithm Development* – Implement core MARL algorithms (MADDPG, QMIX, COMA, PPO, etc.) and ensure they run on simple benchmarks (MPE, PettingZoo). Start small-scale experiments comparing their performance.  
- **Month 6–8:** *Distributed Training Infrastructure* – Integrate Ray RLlib or custom distributed setup. Implement IMPALA-style parallel training. Test scaling (multi-GPU, multi-node). Add hyperparameter tuning.  
- **Month 9–10:** *Environment Expansion* – Connect to complex simulators (Unity 3D environments, Isaac Gym) and train agents in realistic scenarios. Fine-tune environment wrappers for performance.  
- **Month 11–12:** *Evaluation & Refinement* – Run systematic benchmarks and ablations on selected tasks. Perform robustness/safety tests. Optimize performance and write up documentation.  
- **Delivery:** Compile all findings into `implementation-plan.md`, including diagrams, tables, and references.  

**Resources:** We estimate a team of *2–3 ML engineers/researchers* over 6–12 months. Compute needs: access to ~4–8 modern GPUs (NVIDIA A100/V100) and a few dozen CPU cores. If more are available, larger-scale experiments (e.g. thousands of parallel actors) can be conducted. Budget estimates are environment-specific and omitted here. 

## Risk Analysis and Mitigation  

Major risks include:

- **Convergence/Non-stationarity:** Agents may fail to learn stable policies due to non-stationarity. *Mitigation:* Start with smaller agent counts, use CTDE (central critics), and progressively increase complexity.  
- **Compute Constraints:** Large MARL experiments are GPU-intensive. *Mitigation:* Prioritize experiments, use efficient simulators (frame-skip, vector envs), and monitor GPU utilization. Possibly use cloud spot instances or gradient-based methods if needed.  
- **Reproducibility Issues:** RL experiments can be sensitive to seeds. *Mitigation:* Enforce strict seeding, log versions, and repeat key results multiple times.  
- **Integration Bugs:** Complex pipelines (multi-agent + multi-node) can have hard-to-debug errors. *Mitigation:* Maintain modular code with tests; use logging/visualisation to trace failures.  
- **Algorithmic Complexity:** New algorithms (e.g. QMIX, MAPPO) have many hyperparameters. *Mitigation:* Begin with well-known settings from literature; use AutoML tools (Optuna) to manage tuning.  
- **Safety/Ethics:** Agents might learn unsafe or unethical behaviors (e.g. collision courses) in simulations. *Mitigation:* Incorporate safety constraints in reward; perform adversarial testing; enforce “kill-switch” overrides during real-world trials.  

A formal risk matrix will be maintained to track these and other risks (like project delays, team availability).

## Ethical and Legal Considerations  

We will ensure the system aligns with ethical AI practices:  

- **Safety and Fairness:** MARL policies will be checked for *safe* behaviors (no collisions, discrimination, etc.). In competitive tasks, agents should not exploit unintended loopholes. We will consider fairness metrics (e.g. equitable reward distribution) when agents represent humans (e.g. traffic simulations).  
- **Transparency:** Complex MARL policies are opaque; we will log behaviour trajectories and maintain interpretable evaluation (to the extent possible).  
- **Data Privacy:** If any real data (e.g. environment maps or human demonstration traces) are used, we will comply with privacy laws (GDPR, etc.). All synthetic data has no personal information.  
- **Dual Use:** MARL technology could be misused (e.g. military drones). We will follow organizational/industry guidelines on dual-use AI, and restrict application domains appropriately.  
- **Legal Compliance:** Software licenses of dependencies (Gym, PyTorch, Unity) will be respected. If deploying agents (e.g. in robots), relevant safety certifications (ISO, CE) must be considered (though out of scope for this plan).  

Ethical review: we will document potential biases (e.g. simulation simplifications) and consult domain experts (e.g. safety engineers) as needed. 

---

This plan integrates best practices from recent MARL research and engineering.  Primary references (papers and official docs) have informed each design choice. Subsequent sections of the final `implementation-plan.md` will expand these topics in detail, with accompanying tables (above) and *Mermaid* diagrams illustrating architecture and training flow. 

