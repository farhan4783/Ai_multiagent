# STGAT-MARL: Project Startup & Command Manual 🚀⚡

This guide provides the complete, copy-pasteable command instructions to set up, test, and launch the **STGAT-MARL Intelligent Project Management & Dynamic Scheduling System** on your local machine.

---

## ⚡ The 30-Second Express Quickstart

If you already have Python and Node installed, run these 4 commands:

```bash
# 1. Install Python dependencies in editable mode
pip install -e ".[dev,ml]"

# 2. Build the React web dashboard
cd web && npm install && npm run build && cd ..

# 3. Verify the system (runs all 96 unit & integration tests)
pytest -v

# 4. Launch the Mission Control Server & Web Dashboard on Port 8000
python scripts/run_orchestrator_server.py --port 8000
```
Then open your browser to: **`http://127.0.0.1:8000`** 🎉

---

## Table of Contents

1. [Prerequisites Checklist](#1-prerequisites-checklist)
2. [Step 1: Clone Repository & Create Virtual Environment](#step-1-clone-repository--create-virtual-environment)
3. [Step 2: Install Package & Dependencies](#step-2-install-package--dependencies)
4. [Step 3: Build the Web Dashboard](#step-3-build-the-web-dashboard)
5. [Step 4: Verify the Installation with Pytest](#step-4-verify-the-installation-with-pytest)
6. [Step 5: Launch the Orchestration Server & Web UI](#step-5-launch-the-orchestration-server--web-ui)
7. [Step 6: Explore the Web Dashboard](#step-6-explore-the-web-dashboard)
8. [Step 7: Running Specific Subsystem CLI Tools](#step-7-running-specific-subsystem-cli-tools)
   - [7.1 Spatiotemporal Graph Mining](#71-spatiotemporal-graph-mining)
   - [7.2 Google OR-Tools Exact Solver & Baselines](#72-google-or-tools-exact-solver--baselines)
   - [7.3 Multi-Agent RL Swarm Training & Eval](#73-multi-agent-rl-swarm-training--eval)
   - [7.4 Phase 7 Ablation Study Suite](#74-phase-7-ablation-study-suite)
9. [Step 8: Python Programmatic Usage](#step-8-python-programmatic-usage)
10. [Troubleshooting & FAQs](#10-troubleshooting--faqs)

---

## 1. Prerequisites Checklist

Before you begin, ensure you have the following installed:

| Tool | Minimum Version | Check Command |
| :--- | :---: | :--- |
| **Python** | 3.10+ (Supports 3.10, 3.11, 3.12, 3.13, 3.14) | `python --version` or `py --version` |
| **Git** | 2.30+ | `git --version` |
| **Node.js** | 18+ (Includes npm) | `node --version` |
| **Terminal** | PowerShell (Windows) or Bash/Zsh (Linux/macOS) | - |

---

## Step 1: Clone Repository & Create Virtual Environment

### On Windows (PowerShell):
```powershell
# 1. Clone the repository
git clone https://github.com/farhan4783/Ai_multiagent.git
cd Ai_multiagent

# 2. Create Python virtual environment
py -m venv .venv

# 3. Activate virtual environment
.venv\Scripts\Activate.ps1
```

*(Note: If PowerShell complains about script execution policies, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### On Linux / macOS (Bash / Zsh):
```bash
# 1. Clone the repository
git clone https://github.com/farhan4783/Ai_multiagent.git
cd Ai_multiagent

# 2. Create Python virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate
```

---

## Step 2: Install Package & Dependencies

Install `project-marl` in editable development mode along with all machine learning and testing dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,ml]"
```

Or install the core packages directly:
```bash
pip install pydantic networkx numpy scipy gymnasium pettingzoo ortools torch pytest pytest-cov fastapi uvicorn
```

---

## Step 3: Build the Web Dashboard

The web dashboard is built with React 18, Vite, and TypeScript. Build the static bundle so the FastAPI server can serve it:

```bash
# Enter the web directory
cd web

# Install frontend packages
npm install

# Compile production bundle into web/dist
npm run build

# Return to repository root
cd ..
```

---

## Step 4: Verify the Installation with Pytest

Run the comprehensive test suite to confirm that all 8 phases are operational and passing:

```bash
pytest -v
```

Expected output:
```text
====================== 96 passed, 1 warning in 18.53s ======================
```

You can also run specific subsystem test files:
```bash
# Test only the Phase 7 Orchestrator and API
pytest tests/test_orchestrator.py -v

# Test only the Sandboxed Git Worktree Execution Layer
pytest tests/test_workspace.py -v

# Test only the Multi-Agent RL Swarm
pytest tests/test_marl.py -v

# Test only the STGAT Graph Neural Network
pytest tests/test_stgat.py -v
```

---

## Step 5: Launch the Orchestration Server & Web UI

Start the unified FastAPI server that mounts the operator dashboard and opens live WebSocket streaming:

```bash
python scripts/run_orchestrator_server.py --port 8000 --developers 3
```

You will see:
```text
======================================================================
      STGAT-MARL AUTONOMOUS PROJECT ORCHESTRATION SERVER      
======================================================================
  Host:             127.0.0.1
  Port:             8000
  Repository:       C:\...\Ai_multiagent
  Developer Agents: 3
  STGAT Checkpoint: checkpoints/best_stgat.pt
======================================================================

[INFO] Ingesting repository codebase into Spatiotemporal Graph...
[INFO] Analyzed 63 modules.
[INFO] Built Unified Graph: 6 tasks, 63 modules, 1459 edges, Makespan=46.0h
[INFO] Starting Uvicorn server on http://127.0.0.1:8000
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## Step 6: Explore the Web Dashboard

Open your web browser and navigate to:
👉 **`http://127.0.0.1:8000`**

### What to try on the Dashboard:
1. **Graph Tab:** Observe the force-directed 2D canvas of tasks and code files. GATv2 attention weights are rendered as glowing connection lasers.
2. **Timeline (Gantt) Tab:** View the dynamic schedule with Critical Path tasks highlighted in amber.
3. **Agent Swarm Tab:** Inspect each agent's allocated Git worktree path, primary port lease (`24000+`), and live terminal logs.
4. **PromptQL Console Tab:** Type a query like `"critical path"` or `"JWT authentication"` and click **Search Memory / RAG** to test sub-100ms context retrieval.
5. **Trigger Simulation Step:** Click the purple **"Trigger Simulation Step"** button in the header to advance the orchestration loop and observe real-time schedule updates.
6. **Manual Intervention:** Click **"Manual Intervention"** to override a task priority, inject buffer margin, or add an ad-hoc work item.

---

## Step 7: Running Specific Subsystem CLI Tools

### 7.1 Spatiotemporal Graph Mining
Scan the codebase, extract AST call graphs, compute cyclomatic complexity, mine git commit histories for co-churn, calculate Critical Path Method (CPM) metrics, and export the graph tensor:

```bash
python scripts/run_graph_miner.py --repo-dir . --output-json data/graph_state.json
```

---

### 7.2 Google OR-Tools Exact Solver & Baselines
Solve static project instances to mathematical optimality using Google OR-Tools CP-SAT, and benchmark heuristic dispatching policies (CPM, SPT, LPT, MSF, Random):

```bash
python scripts/run_baselines_benchmark.py --episodes 5 --developers 3 --max-tasks 12 --seed 42
```

---

### 7.3 Multi-Agent RL Swarm Training & Eval
Train the decentralized actor swarm and centralized critic using MAPPO over parallel stochastic project episodes:

```bash
# Train MAPPO for 50 episodes with 3 developer agents:
python scripts/run_marl_training.py --episodes 50 --devs 3 --tasks 15 --modules 10

# Evaluate a saved checkpoint across 10 test episodes:
python scripts/run_marl_training.py --eval-only --checkpoint checkpoints/mappo_latest.pt --tasks 8 --devs 2
```

---

### 7.4 Phase 7 Ablation Study Suite
Run the formal multi-dimensional ablation benchmarks evaluating predictive accuracy, multi-agent credit assignment, and large-scale graph scalability up to $N = 500+$:

```bash
python scripts/run_ablation_studies.py
```
Reports will be exported to:
- `data/ablation_benchmark_results.json`
- `data/ablation_report.md`

---

## Step 8: Python Programmatic Usage

You can also use `project_marl` directly as a Python library:

```python
from project_marl.orchestrator.coordinator import AutonomousProjectCoordinator, OrchestratorConfig

# 1. Initialize Coordinator
config = OrchestratorConfig(repo_dir=".", num_developers=3)
coordinator = AutonomousProjectCoordinator(config)

# 2. Ingest Codebase & Synthesize Spatiotemporal Graph
coordinator.ingest_project()

# 3. Execute an Orchestration Step
step_result = coordinator.step()
print("Step Result:", step_result)

# 4. Human-in-the-Loop Intervention
coordinator.apply_intervention(
    intervention_type="PRIORITY_OVERRIDE",
    params={"task_id": "TASK-01", "priority": 5}
)

# 5. Clean teardown
coordinator.reset()
```

---

## 10. Troubleshooting & FAQs

### Q: Port 8000 is already in use by another process.
**Fix:** Start the server on a different port, e.g., 8080:
```bash
python scripts/run_orchestrator_server.py --port 8080
```
Or kill the process occupying port 8000 on Windows PowerShell:
```powershell
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

### Q: PowerShell gives an error: `Activate.ps1 cannot be loaded because running scripts is disabled`.
**Fix:** Run this one-time command in PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Q: The web dashboard loads a blank page or 404 on `http://127.0.0.1:8000`.
**Fix:** Ensure you have compiled the frontend bundle into `web/dist`:
```bash
cd web
npm install
npm run build
cd ..
```

### Q: Git worktrees fail with "fatal: already exists".
**Fix:** Teardown any dangling test worktrees:
```powershell
Remove-Item -Recurse -Force .worktrees
git worktree prune
```

---

**You are all set! Enjoy orchestrating autonomous AI agent swarms with STGAT-MARL!** 🚀
