"""
Comprehensive unit and integration test suite for the ProjectSchedulingEnv
PettingZoo Parallel environment.
"""

import pytest
import numpy as np
import gymnasium as gym
from pettingzoo.test import parallel_api_test

from project_marl.core.constants import ExecutionStatus
from project_marl.core.schemas import TaskNode
from project_marl.sim.project_env import ProjectSchedulingEnv, PM_WEIGHT_PRESETS
from project_marl.sim.generator import SyntheticProjectGenerator


class TestProjectSchedulingEnv:
    """Test suite for the PettingZoo ProjectSchedulingEnv."""

    def test_environment_initialization(self) -> None:
        """Verifies agents, action spaces, and observation spaces are properly constructed."""
        env = ProjectSchedulingEnv(num_developers=3, max_tasks=10, max_steps=50, seed=123)

        expected_agents = [
            "scheduling_agent",
            "risk_agent",
            "developer_agent_0",
            "developer_agent_1",
            "developer_agent_2",
            "pm_agent",
        ]
        assert env.possible_agents == expected_agents
        assert len(env.agents) == 6

        # Check action spaces
        assert isinstance(env.action_space("scheduling_agent"), gym.spaces.Discrete)
        assert env.action_space("scheduling_agent").n == 10 * 3 + 1
        assert isinstance(env.action_space("risk_agent"), gym.spaces.Discrete)
        assert env.action_space("risk_agent").n == 10 + 1
        for dev in env.dev_agent_ids:
            assert isinstance(env.action_space(dev), gym.spaces.Discrete)
            assert env.action_space(dev).n == 3
        assert isinstance(env.action_space("pm_agent"), gym.spaces.Discrete)
        assert env.action_space("pm_agent").n == 4

        # Check observation spaces
        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            assert isinstance(obs_space, gym.spaces.Box)
            assert obs_space.dtype == np.float32

    def test_pettingzoo_parallel_api_conformance(self) -> None:
        """Ensures 100% strict compliance with PettingZoo's standard Parallel API test."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=8, max_steps=25, seed=42)
        parallel_api_test(env, num_cycles=10)

    def test_reset_behavior(self) -> None:
        """Verifies reset initializes project DAG and returns valid observations."""
        env = ProjectSchedulingEnv(num_developers=3, max_tasks=10, seed=42)
        obs, infos = env.reset(seed=42)

        assert set(obs.keys()) == set(env.possible_agents)
        assert set(infos.keys()) == set(env.possible_agents)

        for agent, o in obs.items():
            assert env.observation_space(agent).contains(o)
            assert not np.isnan(o).any()

        assert env.current_time == 0.0
        assert env.step_count == 0
        assert len(env.completed_tasks) == 0
        assert env.num_tasks_actual > 0

    def test_scheduling_dispatch_action(self) -> None:
        """Tests that dispatching a ready task assigns it to the target developer."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        ready_tasks = env._get_ready_tasks()
        assert len(ready_tasks) > 0
        target_task = ready_tasks[0]
        target_task_idx = env.task_list.index(target_task.task_id)

        # Dispatch target_task to dev_0
        sched_action = target_task_idx * env.num_developers + 0
        actions = {
            "scheduling_agent": sched_action,
            "risk_agent": env.max_tasks,  # NO_OP
            "developer_agent_0": 1,        # Balanced
            "developer_agent_1": 1,
            "pm_agent": 1,
        }

        obs, rewards, terms, truncs, infos = env.step(actions)

        # Developer 0 should now be working on target_task
        assert env.dev_current_task[0] == target_task.task_id
        assert env.tasks[target_task.task_id].status in (
            ExecutionStatus.IN_PROGRESS,
            ExecutionStatus.COMPLETED,
            ExecutionStatus.BLOCKED,
        )

    def test_risk_buffer_injection(self) -> None:
        """Tests that RiskAgent can inject buffer hours to pending tasks."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        target_task_idx = 0
        target_task_id = env.task_list[target_task_idx]
        initial_buffer = env.task_buffers[target_task_id]

        actions = {
            "scheduling_agent": env.max_tasks * env.num_developers,  # NO_OP
            "risk_agent": target_task_idx,  # Inject buffer to task 0
            "developer_agent_0": 1,
            "developer_agent_1": 1,
            "pm_agent": 1,
        }

        env.step(actions)
        assert env.task_buffers[target_task_id] == initial_buffer + 2.0

    def test_pm_weight_switching(self) -> None:
        """Tests that PMAgent can dynamically switch reward trade-off weights."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=10, seed=42)
        env.reset(seed=42)

        # Select Quality priority (preset 2)
        actions = {
            "scheduling_agent": env.max_tasks * env.num_developers,
            "risk_agent": env.max_tasks,
            "developer_agent_0": 1,
            "developer_agent_1": 1,
            "pm_agent": 2,
        }

        env.step(actions)
        np.testing.assert_array_equal(env.active_weights, PM_WEIGHT_PRESETS[2])

    def test_render_output(self) -> None:
        """Tests that render returns a descriptive string containing key metrics."""
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=6, seed=42)
        env.reset(seed=42)

        rendered = env.render(mode="ansi")
        assert "Project Scheduling Sim" in rendered
        assert "Developer Worktrees" in rendered
        assert "Dev 0" in rendered
        assert "Ready Tasks" in rendered

    def test_full_episode_completion(self) -> None:
        """Tests stepping until project completion (all tasks complete or max_steps reached)."""
        env = ProjectSchedulingEnv(num_developers=3, max_tasks=6, max_steps=100, seed=10)
        obs, info = env.reset(seed=10)

        step = 0
        done = False
        while not done and step < 100:
            step += 1
            # Simple greedy dispatch: find first ready task and first idle dev
            ready = env._get_ready_tasks()
            idle_devs = [m for m, t in enumerate(env.dev_current_task) if t is None]

            if ready and idle_devs:
                t_idx = env.task_list.index(ready[0].task_id)
                sched_act = t_idx * env.num_developers + idle_devs[0]
            else:
                sched_act = env.max_tasks * env.num_developers

            actions = {
                "scheduling_agent": sched_act,
                "risk_agent": env.max_tasks,
                "developer_agent_0": 1,
                "developer_agent_1": 1,
                "developer_agent_2": 1,
                "pm_agent": 1,
            }

            obs, rewards, terms, truncs, infos = env.step(actions)
            done = all(terms.values()) or all(truncs.values())

        assert done
        assert len(env.agents) == 0  # Standard PettingZoo behavior when episode finishes
