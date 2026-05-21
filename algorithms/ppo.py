"""
Custom PPO implementation for Multi-Agent RL with Centralized Critic.
Supports heterogeneous agents with MultiDiscrete action spaces.
Replaces Ray RLlib's PPO.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helper: SlimFC equivalent
# ---------------------------------------------------------------------------

class SlimFC(nn.Module):
    """Equivalent of RLlib's SlimFC — a linear layer with optional activation."""
    def __init__(self, in_size, out_size, activation_fn=None, initializer=None):
        super().__init__()
        self.linear = nn.Linear(in_size, out_size)
        if initializer is not None:
            try:
                initializer(self.linear.weight)
            except Exception:
                pass
        nn.init.zeros_(self.linear.bias)
        if activation_fn is not None and isinstance(activation_fn, type):
            activation_fn = activation_fn()
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.linear(x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        return x


# ---------------------------------------------------------------------------
# Helper: add_time_dimension (RLlib compatible)
# ---------------------------------------------------------------------------

def add_time_dimension(x, seq_lens, time_major=False):
    """
    Reshape flat [B, F] tensor to [B, T, F] using sequence lengths.
    Simplified version of RLlib's add_time_dimension.
    """
    if seq_lens is None:
        return x.unsqueeze(1)
    B, F = x.shape
    T = int(seq_lens.max().item()) if isinstance(seq_lens, torch.Tensor) else int(max(seq_lens))
    if B % T == 0:
        # All sequences same length — simple reshape
        x = x.reshape(B // T, T, F)
    else:
        # Ragged — pad and mask (simplified: assume complete episodes)
        if B == T:
            x = x.unsqueeze(0)
        else:
            x = x.reshape(B // T, T, F)
    if time_major:
        x = x.transpose(0, 1)
    return x


# ---------------------------------------------------------------------------
# Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self):
        self.obs = []          # list of dicts (augmented observations)
        self.actions = []      # list of tensors
        self.log_probs = []    # list of scalars
        self.values = []       # list of scalars
        self.rewards = []      # list of scalars
        self.dones = []        # list of bools
        self.states = []       # list of recurrent states (if any)
        self.seq_lens = []     # list of episode lengths (for recurrent models)
        self.advantages = []   # pre-computed GAE advantages
        self.returns = []      # pre-computed GAE returns

    def add(self, obs, action, log_prob, value, reward, done, state=None):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.states.append(state)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.states.clear()
        self.seq_lens.clear()
        self.advantages.clear()
        self.returns.clear()

    def set_gae(self, advantages, returns):
        """Set pre-computed GAE advantages and returns."""
        self.advantages = advantages
        self.returns = returns

    def size(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPOPolicy — single agent policy
# ---------------------------------------------------------------------------

class PPOPolicy:
    """
    Wraps a model for PPO training.
    Handles action sampling for MultiDiscrete action spaces.
    """

    def __init__(self, model, action_dims, lr=1e-4, clip_param=0.25,
                 entropy_coeff=0.01, vf_coeff=0.5, max_grad_norm=0.5):
        self.model = model
        self.action_dims = action_dims  # list of n_categories per action dim
        self.clip_param = clip_param
        self.entropy_coeff = entropy_coeff
        self.vf_coeff = vf_coeff
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def _split_logits(self, logits):
        """Split flat logits into per-dimension logits for MultiDiscrete."""
        return torch.split(logits, self.action_dims, dim=-1)

    def sample_actions(self, logits, explore=True):
        """Sample actions from MultiDiscrete logits. Returns (actions, log_prob, entropy)."""
        splits = self._split_logits(logits)
        actions = []
        log_probs = []
        entropies = []
        for logit in splits:
            cat = Categorical(logits=logit)
            if explore:
                act = cat.sample()
            else:
                act = cat.probs.argmax(dim=-1)
            actions.append(act)
            log_probs.append(cat.log_prob(act))
            entropies.append(cat.entropy())
        actions = torch.stack(actions, dim=-1)
        total_log_prob = torch.sum(torch.stack(log_probs, dim=-1), dim=-1)
        total_entropy = torch.sum(torch.stack(entropies, dim=-1), dim=-1)
        return actions, total_log_prob, total_entropy

    def evaluate_actions(self, obs_batch, actions_batch, states=None, seq_lens=None):
        """
        Evaluate actions for PPO update.
        Returns: (new_log_probs, entropy, values, new_states)
        """
        logits, new_states = self.model({"obs": obs_batch}, states, seq_lens)
        values = self.model.value_function()
        splits = self._split_logits(logits)
        new_log_probs = []
        entropies = []
        for i, logit in enumerate(splits):
            cat = Categorical(logits=logit)
            act = actions_batch[:, i]
            new_log_probs.append(cat.log_prob(act))
            entropies.append(cat.entropy())
        new_log_probs = torch.sum(torch.stack(new_log_probs, dim=-1), dim=-1)
        entropy = torch.mean(torch.stack(entropies, dim=-1))
        return new_log_probs, entropy, values, new_states

    def compute_single_action(self, obs, state=None, explore=False):
        """Compute action for a single observation (evaluation)."""
        with torch.no_grad():
            obs_t = self._obs_to_tensor(obs)
            if state is not None and not isinstance(state, list):
                state = [state]
            logits, new_state = self.model(
                {"obs": obs_t}, state, seq_lens=torch.tensor([1]))
            value = self.model.value_function()
            actions, _, _ = self.sample_actions(logits, explore=explore)
        action = actions.squeeze(0).cpu().numpy()
        if new_state is not None and len(new_state) > 0:
            new_state = [s.cpu() for s in new_state]
        return action, new_state, value.item()

    def _obs_to_tensor(self, obs):
        """Convert observation dict (numpy) to tensor dict."""
        if isinstance(obs, dict):
            return {k: torch.from_numpy(v).unsqueeze(0).float() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()}
        return torch.from_numpy(obs).unsqueeze(0).float()

    def update(self, buffer, num_epochs=10, mini_batch_size=256):
        """PPO update on collected rollout data. Uses pre-computed GAE from buffer."""
        if buffer.size() == 0:
            return {}

        n = buffer.size()
        obs_list = buffer.obs
        act_list = buffer.actions
        old_lp_list = buffer.log_probs
        adv_list = buffer.advantages
        ret_list = buffer.returns

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        indices = np.arange(n)

        for epoch in range(num_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, mini_batch_size):
                batch_idx = indices[start:start + mini_batch_size]
                batch_obs = self._batch_obs([obs_list[i] for i in batch_idx])
                batch_acts = torch.stack([act_list[i] for i in batch_idx])
                batch_old_lp = torch.stack([old_lp_list[i] for i in batch_idx])
                batch_adv = torch.tensor([adv_list[i] for i in batch_idx], dtype=torch.float32)
                batch_ret = torch.tensor([ret_list[i] for i in batch_idx], dtype=torch.float32)

                # Get seq_lens for this batch (if available)
                seq_lens = None

                # Evaluate actions
                new_log_probs, entropy, values, _ = self.evaluate_actions(
                    batch_obs, batch_acts, states=None, seq_lens=seq_lens)

                values = values.squeeze(-1)

                # PPO clip loss
                ratio = torch.exp(new_log_probs - batch_old_lp)
                adv_norm = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
                surr1 = ratio * adv_norm
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * adv_norm
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(values, batch_ret)

                # Total loss
                loss = policy_loss + self.vf_coeff * value_loss - self.entropy_coeff * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item() if isinstance(entropy, torch.Tensor) else entropy
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def _batch_obs(self, obs_list):
        """Stack a list of observation dicts into a batch dict."""
        if isinstance(obs_list[0], dict):
            batched = {}
            for key in obs_list[0].keys():
                vals = [o[key] for o in obs_list]
                if isinstance(vals[0], torch.Tensor):
                    batched[key] = torch.stack(vals)
                else:
                    batched[key] = torch.tensor(np.stack(vals), dtype=torch.float32)
            return batched
        return torch.stack(obs_list)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def export_model(self, path):
        """Export model state_dict for self-play inference."""
        torch.save(self.model, path)


# ---------------------------------------------------------------------------
# MultiAgentPPO — multi-agent PPO with centralized critic
# ---------------------------------------------------------------------------

class MultiAgentPPO:
    """
    Multi-Agent PPO supporting heterogeneous agents with centralized critic.
    Compatible with the HHMARL project's training scripts.
    """

    def __init__(self, policies, env_fn, train_batch_size=2000, mini_batch_size=256,
                 gamma=0.99, lambda_=0.95, num_sgd_iter=10,
                 observation_fn=None, postprocess_fn=None):
        """
        Args:
            policies: dict of {policy_id: PPOPolicy}
            env_fn: callable that returns a new environment instance
            train_batch_size: total timesteps per training iteration
            mini_batch_size: SGD mini-batch size
            gamma: discount factor
            lambda_: GAE lambda
            num_sgd_iter: number of SGD epochs per update
            observation_fn: function to augment observations for centralized critic
            postprocess_fn: function to post-process trajectory (add actual actions)
        """
        self.policies = policies
        self.env_fn = env_fn
        self.train_batch_size = train_batch_size
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_sgd_iter = num_sgd_iter
        self.observation_fn = observation_fn
        self.postprocess_fn = postprocess_fn

        self.logdir = None
        self._timesteps = 0
        self._episodes = 0

    def _make_env(self):
        """Create a fresh env instance."""
        return self.env_fn()

    def train(self):
        """
        One training iteration:
        1. Collect rollouts until train_batch_size transitions
        2. Update each policy with PPO
        Returns dict with training stats.
        """
        # Collect rollouts
        buffers = defaultdict(RolloutBuffer)
        total_reward = 0.0
        total_episodes = 0
        total_steps = 0

        env = self._make_env()

        while total_steps < self.train_batch_size:
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            ep_steps = 0
            episode_data = defaultdict(list)  # raw episode data per policy
            recurrent_states = {
                pid: pol.model.get_initial_state() if hasattr(pol.model, 'get_initial_state') else None
                for pid, pol in self.policies.items()
            }

            while not done:
                actions = {}
                aug_obs = {}

                for agent_id in obs.keys():
                    policy_id = f"ac{agent_id}_policy" if f"ac{agent_id}_policy" in self.policies else "commander_policy"

                    # Augment observation for centralized critic
                    if self.observation_fn is not None:
                        all_aug = self.observation_fn(obs.copy())
                        aug = all_aug.get(agent_id, obs)
                    else:
                        aug = {agent_id: obs[agent_id]}

                    aug_obs[agent_id] = aug

                    # Convert to tensor
                    obs_tensor = {k: torch.from_numpy(v).unsqueeze(0).float() if isinstance(v, np.ndarray) else v
                                  for k, v in aug.items()}

                    policy = self.policies[policy_id]
                    state = recurrent_states.get(policy_id)
                    with torch.no_grad():
                        logits, new_state = policy.model(
                            {"obs": obs_tensor},
                            state=[s.unsqueeze(0) for s in state] if state and isinstance(state, list) else None,
                            seq_lens=torch.tensor([1]))
                        value = policy.model.value_function()
                    a, log_prob, entropy = policy.sample_actions(logits, explore=True)

                    episode_data[policy_id].append({
                        "agent_id": agent_id,
                        "aug_obs": aug,
                        "action": a.squeeze(0),
                        "log_prob": log_prob.squeeze(0),
                        "value": value.squeeze(0),
                        "state": [s.clone() for s in new_state] if new_state else None,
                    })
                    if new_state:
                        recurrent_states[policy_id] = [s.squeeze(0) for s in new_state]

                    actions[agent_id] = a.squeeze(0).cpu().numpy()

                # Step environment
                next_obs, rews, terms, truncs, info = env.step(actions)
                done = terms.get("__all__", False) or truncs.get("__all__", False)

                # Assign rewards — match by agent_id
                for pid in episode_data:
                    for t in episode_data[pid]:
                        t["reward"] = 0.0
                for ag_id, r in rews.items():
                    pid = f"ac{ag_id}_policy" if f"ac{ag_id}_policy" in self.policies else "commander_policy"
                    if pid in episode_data:
                        for t in reversed(episode_data[pid]):
                            if t.get("agent_id") == ag_id:
                                t["reward"] = r
                                break
                    ep_reward += r

                ep_steps += 1
                total_steps += 1

                if done or total_steps >= self.train_batch_size:
                    # Post-process episode: add actual actions to augmented obs
                    if self.postprocess_fn is not None:
                        self.postprocess_fn(episode_data)

                    # Compute GAE per policy and add to buffers
                    for pid, traj in episode_data.items():
                        if len(traj) == 0:
                            continue
                        buf = buffers[pid]
                        # Get last value for GAE
                        last_val = 0.0 if done else traj[-1]["value"].item()
                        values = [t["value"].item() for t in traj]
                        rewards = [t["reward"] for t in traj]
                        dones_per_step = [False] * len(traj)
                        if done:
                            dones_per_step[-1] = True

                        # Compute GAE
                        advs, rets = self._compute_gae(rewards, values, dones_per_step, last_val)

                        for i, t in enumerate(traj):
                            buf.obs.append(t["aug_obs"])
                            buf.actions.append(t["action"])
                            buf.log_probs.append(t["log_prob"])
                            buf.values.append(t["value"])
                            buf.rewards.append(t["reward"])
                            buf.dones.append(dones_per_step[i])
                            buf.advantages.append(advs[i])
                            buf.returns.append(rets[i])

                        # Store episode seq_len for recurrent models
                        buf.seq_lens.append(len(traj))

                    total_reward += ep_reward
                    total_episodes += 1
                    break

                obs = next_obs

        self._timesteps += total_steps
        self._episodes += total_episodes

        # Update each policy
        update_stats = {}
        for pid, buf in buffers.items():
            if buf.size() == 0:
                continue
            stats = self.policies[pid].update(buf, num_epochs=self.num_sgd_iter,
                                              mini_batch_size=self.mini_batch_size)
            for k, v in stats.items():
                update_stats[f"{pid}_{k}"] = v

        avg_reward = total_reward / max(total_episodes, 1)
        update_stats["episode_reward_mean"] = avg_reward
        update_stats["episodes"] = total_episodes
        update_stats["timesteps"] = total_steps

        return update_stats

    def _compute_gae(self, rewards, values, dones, last_value):
        """Compute GAE advantages and returns."""
        advs = []
        rets = []
        gae = 0.0
        vals = values + [last_value]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * vals[t + 1] * (1 - int(dones[t])) - vals[t]
            gae = delta + self.gamma * self.lambda_ * (1 - int(dones[t])) * gae
            advs.insert(0, gae)
            rets.insert(0, gae + vals[t])
        return advs, rets

    def save(self, checkpoint_dir=None):
        """Save all policies to checkpoint directory."""
        if checkpoint_dir is None:
            checkpoint_dir = self.logdir
        os.makedirs(checkpoint_dir, exist_ok=True)
        for pid, policy in self.policies.items():
            sd = policy.state_dict()
            torch.save(sd, os.path.join(checkpoint_dir, f"{pid}.pt"))
        # Save training state
        meta = {"timesteps": self._timesteps, "episodes": self._episodes}
        torch.save(meta, os.path.join(checkpoint_dir, "meta.pt"))

    def restore(self, checkpoint_dir):
        """Restore all policies from checkpoint directory."""
        for pid, policy in self.policies.items():
            path = os.path.join(checkpoint_dir, f"{pid}.pt")
            if os.path.exists(path):
                sd = torch.load(path, map_location="cpu")
                policy.load_state_dict(sd)
        meta_path = os.path.join(checkpoint_dir, "meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, map_location="cpu")
            self._timesteps = meta.get("timesteps", 0)
            self._episodes = meta.get("episodes", 0)

    def export_policy_model(self, dir_path, policy_id):
        """Export a single policy model for self-play inference."""
        os.makedirs(dir_path, exist_ok=True)
        policy = self.policies[policy_id]
        torch.save(policy.model, os.path.join(dir_path, "model.pt"))

    def compute_single_action(self, observation, state=None, policy_id=None, explore=False):
        """
        Compute action for a single observation (evaluation).
        RLlib-compatible interface.
        Returns: (action_array, [state0, state1], {})
        """
        if policy_id is None:
            policy_id = list(self.policies.keys())[0]
        policy = self.policies[policy_id]

        if state is not None and not isinstance(state, list):
            state = [state]

        action, new_state, value = policy.compute_single_action(observation, state, explore)
        if new_state is None:
            new_state = []
        return action, new_state, {}
