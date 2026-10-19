"""
Comprehensive Unit and Integration Test Suite for Phase 4: Multi-Agent Reinforcement Learning (MARL).

Tests:
    1. Actor Policies (SchedulingActor, RiskActor, DeveloperActor, PMActor) forward pass and sampling.
    2. Action masking enforcement (invalid actions strictly have 0 probability).
    3. MultiAgentActorSwarm joint coordination and action evaluation.
    4. Centralized Critic V_phi(S_t) value estimation and gradient flow.
    5. RoleSpecificCentralizedCritic and CentralizedSTGATCritic fusion.
    6. MAPPORolloutBuffer insertion, GAE advantage calculation, and batch generation.
    7. MAPPOTrainer optimization step and metrics accounting.
    8. QMIX mixing network monotonicity constraint verification (dQ_tot / dQ_i >= 0).
    9. QMIXTrainer optimization step with multi-agent replay buffer.
    10. End-to-end environment rollout loop with ProjectSchedulingEnv.
    11. Checkpoint serialization and deserialization.
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from project_marl.sim.project_env import ProjectSchedulingEnv
from project_marl.agents.policies import (
    BaseActorPolicy,
    SchedulingActor,
    RiskActor,
    DeveloperActor,
    PMActor,
    MultiAgentActorSwarm,
)
from project_marl.agents.critics import (
    CentralizedCritic,
    RoleSpecificCentralizedCritic,
    CentralizedSTGATCritic,
)
from project_marl.agents.mappo import (
    MAPPORolloutBuffer,
    MAPPOTrainer,
)
from project_marl.agents.qmix import (
    AgentQNetwork,
    QMixer,
    MultiAgentReplayBuffer,
    QMIXTrainer,
)


class TestMARLPolicies:
    """Tests for actor policies and action masking."""

    def test_scheduling_actor_shapes_and_sampling(self) -> None:
        obs_dim = 64
        max_tasks = 10
        num_devs = 2
        actor = SchedulingActor(obs_dim=obs_dim, max_tasks=max_tasks, num_developers=num_devs)

        obs = torch.randn(4, obs_dim)
        dist = actor(obs)
        assert dist.probs.shape == (4, max_tasks * num_devs + 1)

        action, log_prob, entropy = actor.sample(obs)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)

    def test_action_masking_strictly_enforced(self) -> None:
        """Verifies that masked actions have zero probability and are never sampled."""
        obs_dim = 32
        max_tasks = 5
        num_devs = 2
        actor = SchedulingActor(obs_dim=obs_dim, max_tasks=max_tasks, num_developers=num_devs)

        # Allow ONLY action 3 and NO_OP (action 10)
        action_dim = max_tasks * num_devs + 1
        mask = torch.zeros(action_dim)
        mask[3] = 1.0
        mask[actor.noop_action] = 1.0

        obs = torch.randn(obs_dim)
        dist = actor(obs, action_mask=mask)

        probs = dist.probs.squeeze()
        for a in range(action_dim):
            if a in (3, actor.noop_action):
                assert probs[a].item() > 0.0
            else:
                assert probs[a].item() == pytest.approx(0.0, abs=1e-6)

        # Sample 50 times: should never sample any other action
        for _ in range(50):
            sampled, _, _ = actor.sample(obs, action_mask=mask)
            assert sampled.item() in (3, actor.noop_action)

    def test_developer_and_pm_actors(self) -> None:
        dev_actor = DeveloperActor(obs_dim=16)
        pm_actor = PMActor(obs_dim=13)

        dev_obs = torch.randn(16)
        pm_obs = torch.randn(13)

        dev_act, dev_lp, _ = dev_actor.sample(dev_obs)
        assert 0 <= dev_act.item() <= 2
        assert not torch.isnan(dev_lp)

        pm_act, pm_lp, _ = pm_actor.sample(pm_obs)
        assert 0 <= pm_act.item() <= 3
        assert not torch.isnan(pm_lp)

    def test_actor_swarm_act_and_evaluate(self) -> None:
        swarm = MultiAgentActorSwarm(
            sched_obs_dim=60,
            risk_obs_dim=40,
            dev_obs_dim=16,
            pm_obs_dim=13,
            num_developers=2,
            max_tasks=10,
        )

        obs_dict = {
            "scheduling_agent": np.random.randn(60).astype(np.float32),
            "risk_agent": np.random.randn(40).astype(np.float32),
            "developer_agent_0": np.random.randn(16).astype(np.float32),
            "developer_agent_1": np.random.randn(16).astype(np.float32),
            "pm_agent": np.random.randn(13).astype(np.float32),
        }

        actions, log_probs, entropies = swarm.act(obs_dict)
        assert set(actions.keys()) == set(obs_dict.keys())
        assert set(log_probs.keys()) == set(obs_dict.keys())

        # Test batch action evaluation
        ag_obs = torch.randn(8, 60)
        ag_acts = torch.randint(0, 21, (8,))
        lps, ents = swarm.evaluate_agent_actions("scheduling_agent", ag_obs, ag_acts)
        assert lps.shape == (8,)
        assert ents.shape == (8,)


class TestMARLCritics:
    """Tests for Centralized Critic architectures."""

    def test_centralized_critic_forward_and_backward(self) -> None:
        state_dim = 120
        critic = CentralizedCritic(state_dim=state_dim, hidden_dim=128, num_res_blocks=2)

        states = torch.randn(16, state_dim)
        values = critic(states)
        assert values.shape == (16, 1)

        target = torch.randn(16, 1)
        loss = torch.nn.functional.mse_loss(values, target)
        loss.backward()

        for p in critic.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert not torch.isnan(p.grad).any()

    def test_role_specific_centralized_critic(self) -> None:
        state_dim = 64
        agents = ["scheduling_agent", "risk_agent", "developer_agent_0"]
        critic = RoleSpecificCentralizedCritic(state_dim=state_dim, agent_ids=agents, hidden_dim=64)

        states = torch.randn(8, state_dim)
        val_dict = critic(states)
        assert set(val_dict.keys()) == set(agents)
        for ag in agents:
            assert val_dict[ag].shape == (8, 1)

    def test_centralized_stgat_critic(self) -> None:
        critic = CentralizedSTGATCritic(graph_embed_dim=64, resource_state_dim=32, hidden_dim=64)
        graph_embed = torch.randn(4, 64)
        resource_state = torch.randn(4, 32)
        v = critic(graph_embed, resource_state)
        assert v.shape == (4, 1)


class TestMAPPOAlgorithm:
    """Tests for MAPPO rollout buffer, GAE, and training updates."""

    def test_rollout_buffer_gae_and_batches(self) -> None:
        buffer_size = 20
        state_dim = 30
        agent_obs_dims = {"scheduling_agent": 25, "developer_agent_0": 10}
        agent_action_dims = {"scheduling_agent": 15, "developer_agent_0": 3}

        buf = MAPPORolloutBuffer(
            buffer_size=buffer_size,
            state_dim=state_dim,
            agent_obs_dims=agent_obs_dims,
            agent_action_dims=agent_action_dims,
        )

        for i in range(buffer_size):
            state = np.random.randn(state_dim).astype(np.float32)
            obs = {ag: np.random.randn(agent_obs_dims[ag]).astype(np.float32) for ag in agent_obs_dims}
            actions = {"scheduling_agent": 2, "developer_agent_0": 1}
            lps = {"scheduling_agent": -1.2, "developer_agent_0": -0.8}
            val = float(np.random.randn())
            r = float(np.random.randn())
            done = (i == buffer_size - 1)

            buf.insert(state, obs, actions, lps, val, r, done)

        assert buf.step_idx == buffer_size
        buf.compute_returns_and_advantages(last_value=0.0, last_done=True)

        # Check normalization
        assert pytest.approx(buf.advantages.mean().item(), abs=1e-5) == 0.0
        assert pytest.approx(buf.advantages.std().item(), abs=1e-2) == 1.0

        batches = list(buf.get_batches(batch_size=8))
        assert len(batches) == 3  # 20 / 8 -> 8, 8, 4
        assert batches[0]["states"].shape[0] == 8
        assert batches[2]["states"].shape[0] == 4

    def test_mappo_trainer_update_step(self) -> None:
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=6, max_steps=20, seed=123)
        obs, _ = env.reset(seed=123)

        sched_obs_dim = env.observation_space("scheduling_agent").shape[0]
        risk_obs_dim = env.observation_space("risk_agent").shape[0]
        dev_obs_dim = env.observation_space("developer_agent_0").shape[0]
        pm_obs_dim = env.observation_space("pm_agent").shape[0]
        state_dim = env.state_size

        swarm = MultiAgentActorSwarm(
            sched_obs_dim=sched_obs_dim,
            risk_obs_dim=risk_obs_dim,
            dev_obs_dim=dev_obs_dim,
            pm_obs_dim=pm_obs_dim,
            num_developers=2,
            max_tasks=6,
        )
        critic = CentralizedCritic(state_dim=state_dim, hidden_dim=64)

        trainer = MAPPOTrainer(
            swarm=swarm,
            critic=critic,
            batch_size=16,
            ppo_epochs=2,
        )

        agent_obs_dims = {
            "scheduling_agent": sched_obs_dim,
            "risk_agent": risk_obs_dim,
            "developer_agent_0": dev_obs_dim,
            "developer_agent_1": dev_obs_dim,
            "pm_agent": pm_obs_dim,
        }
        agent_action_dims = {ag: env.action_space(ag).n for ag in env.possible_agents}

        buf = MAPPORolloutBuffer(
            buffer_size=24,
            state_dim=state_dim,
            agent_obs_dims=agent_obs_dims,
            agent_action_dims=agent_action_dims,
        )

        for _ in range(24):
            masks = env.get_action_masks()
            acts, lps, _ = swarm.act(obs, masks)
            val = critic.evaluate_state(env.state())
            next_obs, rewards, terms, truncs, _ = env.step(acts)
            mean_r = float(np.mean(list(rewards.values())))
            done = any(terms.values()) or any(truncs.values())

            buf.insert(env.state(), obs, acts, lps, val, mean_r, done, masks)
            obs = next_obs
            if done:
                obs, _ = env.reset()

        buf.compute_returns_and_advantages(last_value=0.0, last_done=False)
        metrics = trainer.train_step(buf)

        assert "actor_loss" in metrics
        assert "critic_loss" in metrics
        assert not np.isnan(metrics["actor_loss"])
        assert not np.isnan(metrics["critic_loss"])


class TestQMIXAlgorithm:
    """Tests for QMIX value decomposition and hypernetwork monotonicity."""

    def test_qmix_monotonicity_constraint(self) -> None:
        """
        Mathematical proof verification of QMIX monotonicity:
        Ensures dQ_tot / dQ_i >= 0 for all agents and all states.
        """
        num_agents = 4
        state_dim = 20
        mixer = QMixer(num_agents=num_agents, state_dim=state_dim, mixing_embed_dim=16)

        batch_size = 10
        state = torch.randn(batch_size, state_dim)
        agent_qs = torch.randn(batch_size, num_agents, requires_grad=True)

        q_tot = mixer(agent_qs, state)
        assert q_tot.shape == (batch_size, 1)

        # Compute gradient of sum(q_tot) with respect to agent_qs
        grads = torch.autograd.grad(q_tot.sum(), agent_qs)[0]

        # Monotonicity check: all gradients must be non-negative
        assert (grads >= -1e-6).all(), f"Monotonicity violated! Min grad: {grads.min().item()}"

    def test_qmix_trainer_step(self) -> None:
        agents = ["scheduling_agent", "developer_agent_0"]
        state_dim = 30
        agent_obs_dims = {"scheduling_agent": 25, "developer_agent_0": 10}
        agent_action_dims = {"scheduling_agent": 12, "developer_agent_0": 3}

        trainer = QMIXTrainer(
            agents=agents,
            state_dim=state_dim,
            agent_obs_dims=agent_obs_dims,
            agent_action_dims=agent_action_dims,
            mixing_embed_dim=16,
        )

        buffer = MultiAgentReplayBuffer(capacity=100, agents=agents)

        for _ in range(32):
            s = np.random.randn(state_dim).astype(np.float32)
            ns = np.random.randn(state_dim).astype(np.float32)
            obs = {ag: np.random.randn(agent_obs_dims[ag]).astype(np.float32) for ag in agents}
            next_obs = {ag: np.random.randn(agent_obs_dims[ag]).astype(np.float32) for ag in agents}
            acts = {"scheduling_agent": 1, "developer_agent_0": 0}
            masks = {ag: np.ones(agent_action_dims[ag], dtype=np.float32) for ag in agents}
            buffer.push(s, ns, obs, next_obs, acts, 1.0, False, masks, masks)

        batch = buffer.sample(16)
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert not np.isnan(loss)


class TestMARLEndToEnd:
    """Integration test running MAPPO on ProjectSchedulingEnv."""

    def test_marl_environment_interaction_loop(self) -> None:
        env = ProjectSchedulingEnv(num_developers=2, max_tasks=6, max_steps=25, seed=42)
        obs, _ = env.reset(seed=42)

        sched_obs_dim = env.observation_space("scheduling_agent").shape[0]
        risk_obs_dim = env.observation_space("risk_agent").shape[0]
        dev_obs_dim = env.observation_space("developer_agent_0").shape[0]
        pm_obs_dim = env.observation_space("pm_agent").shape[0]

        swarm = MultiAgentActorSwarm(
            sched_obs_dim=sched_obs_dim,
            risk_obs_dim=risk_obs_dim,
            dev_obs_dim=dev_obs_dim,
            pm_obs_dim=pm_obs_dim,
            num_developers=2,
            max_tasks=6,
        )

        total_steps = 0
        done = False
        while not done and total_steps < 30:
            masks = env.get_action_masks()
            actions, _, _ = swarm.act(obs, masks)
            obs, rewards, terms, truncs, _ = env.step(actions)
            total_steps += 1
            done = any(terms.values()) or any(truncs.values())

        assert total_steps > 0
        assert len(env.completed_tasks) <= env.num_tasks_actual

    def test_checkpoint_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "test_mappo.pt")
            swarm = MultiAgentActorSwarm(
                sched_obs_dim=20,
                risk_obs_dim=15,
                dev_obs_dim=10,
                pm_obs_dim=8,
                num_developers=1,
                max_tasks=4,
            )
            critic = CentralizedCritic(state_dim=25, hidden_dim=32)
            trainer = MAPPOTrainer(swarm=swarm, critic=critic)

            trainer.save_checkpoint(ckpt_path)
            assert os.path.exists(ckpt_path)

            # Create fresh trainer and load checkpoint
            new_swarm = MultiAgentActorSwarm(
                sched_obs_dim=20,
                risk_obs_dim=15,
                dev_obs_dim=10,
                pm_obs_dim=8,
                num_developers=1,
                max_tasks=4,
            )
            new_critic = CentralizedCritic(state_dim=25, hidden_dim=32)
            new_trainer = MAPPOTrainer(swarm=new_swarm, critic=new_critic)
            new_trainer.load_checkpoint(ckpt_path)

            # Check weights match
            for p1, p2 in zip(swarm.parameters(), new_swarm.parameters()):
                assert torch.allclose(p1, p2)
            for p1, p2 in zip(critic.parameters(), new_critic.parameters()):
                assert torch.allclose(p1, p2)
