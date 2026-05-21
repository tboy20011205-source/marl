"""
Main file for training the high-level commander policy.
Uses custom PPO implementation instead of Ray RLlib.
"""

import os
import time
import shutil
import tqdm
import torch
import logging
import numpy as np
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from algorithms.ppo import MultiAgentPPO, PPOPolicy, SlimFC
from models.torch_models_hier import CommanderGru
from config import Config
from envs.env_hier import HighLevelEnv

N_OPP_HL = 2  # change for sensing
ACT_DIM = N_OPP_HL + 1
OBS_DIM = 14 + 10 * N_OPP_HL


def update_logs(args):
    """Save checkpoints to experiment log directory."""
    result_dir = os.path.join(args.log_path, 'checkpoint')
    try:
        shutil.rmtree(result_dir)
    except Exception:
        pass
    check_dir = os.path.join(args.log_path, 'checkpoint')
    if os.path.exists(check_dir):
        shutil.copytree(check_dir, result_dir, symlinks=False, dirs_exist_ok=False)


def evaluate(args, algo, env, epoch):
    """Evaluate commander policy and save combat scenario pictures."""

    def cc_obs(obs):
        return {
            "obs_1_own": obs,
            "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
            "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
            "act_1_own": np.zeros(1),
            "act_2": np.zeros(1),
            "act_3": np.zeros(1),
        }

    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    states = [torch.zeros(200), torch.zeros(200)]

    while not done:
        actions = {}
        for ag_id, ag_s in state.items():
            a = algo.compute_single_action(
                observation=cc_obs(ag_s),
                state=states,
                policy_id="commander_policy",
                explore=False)
            actions[ag_id] = a[0]
            if a[1] and len(a[1]) >= 2:
                states[0] = a[1][0]
                states[1] = a[1][1]

        state, rew, hist, trunc, _ = env.step(actions)
        done = hist["__all__"] or trunc["__all__"]
        for r in rew.values():
            reward += r
        step += 1

        if args.render:
            env.plot(Path(args.log_path, "current.png"))
            time.sleep(0.18)

    reward = round(reward, 3)
    env.plot(Path(args.log_path,
                  f"Ep_{epoch}_It_{step}_Rew_{reward}.png"))


def make_checkpoint(args, algo, epoch, env=None):
    algo.save(os.path.join(args.log_path, 'checkpoint'))
    update_logs(args)
    if args.eval and epoch % 500 == 0:
        for _ in range(2):
            evaluate(args, algo, env, epoch)


def get_policy(args):
    """
    Create MultiAgentPPO with the Commander policy.
    Uses centralized critic: each agent sees all agents' observations.
    """

    def central_critic_observer(agent_obs):
        """Augment observations for centralized critic (3 agents)."""
        return {
            1: {
                "obs_1_own": agent_obs[1],
                "obs_2": agent_obs[2],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1, dtype=np.float32),
                "act_2": np.zeros(1, dtype=np.float32),
                "act_3": np.zeros(1, dtype=np.float32),
            },
            2: {
                "obs_1_own": agent_obs[2],
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[3],
                "act_1_own": np.zeros(1, dtype=np.float32),
                "act_2": np.zeros(1, dtype=np.float32),
                "act_3": np.zeros(1, dtype=np.float32),
            },
            3: {
                "obs_1_own": agent_obs[3],
                "obs_2": agent_obs[1],
                "obs_3": agent_obs[2],
                "act_1_own": np.zeros(1, dtype=np.float32),
                "act_2": np.zeros(1, dtype=np.float32),
                "act_3": np.zeros(1, dtype=np.float32),
            },
        }

    def postprocess_trajectory(episode_data):
        """
        Fill in actual actions (normalized by /N_OPP_HL) into augmented obs.
        Equivalent to RLlib CustomCallback.on_postprocess_trajectory.
        Each step has num_agents transitions in agent id order (1, 2, 3).
        """
        for pid, traj in episode_data.items():
            if len(traj) == 0:
                continue
            n_agents = args.num_agents
            num_steps = len(traj) // n_agents

            for s in range(num_steps):
                base = s * n_agents
                step_acts = []
                for a in range(n_agents):
                    act = traj[base + a]["action"].cpu().numpy()
                    step_acts.append(float(act) / N_OPP_HL)

                # Agent 1: own=acts[0], other2=acts[1], other3=acts[2]
                traj[base + 0]["aug_obs"]["act_1_own"] = np.atleast_1d(step_acts[0]).astype(np.float32)
                traj[base + 0]["aug_obs"]["act_2"] = np.atleast_1d(step_acts[1]).astype(np.float32)
                traj[base + 0]["aug_obs"]["act_3"] = np.atleast_1d(step_acts[2]).astype(np.float32)

                # Agent 2: own=acts[1], other2=acts[0], other3=acts[2]
                traj[base + 1]["aug_obs"]["act_1_own"] = np.atleast_1d(step_acts[1]).astype(np.float32)
                traj[base + 1]["aug_obs"]["act_2"] = np.atleast_1d(step_acts[0]).astype(np.float32)
                traj[base + 1]["aug_obs"]["act_3"] = np.atleast_1d(step_acts[2]).astype(np.float32)

                # Agent 3: own=acts[2], other2=acts[0], other3=acts[1]
                traj[base + 2]["aug_obs"]["act_1_own"] = np.atleast_1d(step_acts[2]).astype(np.float32)
                traj[base + 2]["aug_obs"]["act_2"] = np.atleast_1d(step_acts[0]).astype(np.float32)
                traj[base + 2]["aug_obs"]["act_3"] = np.atleast_1d(step_acts[1]).astype(np.float32)

    # Create shared layer
    shared_layer = SlimFC(500, 500, activation_fn=torch.nn.Tanh,
                          initializer=torch.nn.init.orthogonal_)

    # Create Commander model
    model = CommanderGru()
    model.shared_layer = shared_layer

    # Create policy (shared across all agents via policy_mapping_fn pattern)
    policy = PPOPolicy(
        model=model,
        action_dims=[ACT_DIM],  # Discrete action space
        lr=1e-4,
        clip_param=0.25,
        entropy_coeff=0.01,
        vf_coeff=0.5,
        max_grad_norm=0.5,
    )

    policies = {"commander_policy": policy}

    def env_fn():
        return HighLevelEnv(args.env_config)

    algo = MultiAgentPPO(
        policies=policies,
        env_fn=env_fn,
        train_batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gamma=0.99,
        lambda_=0.95,
        num_sgd_iter=10,
        observation_fn=central_critic_observer,
        postprocess_fn=postprocess_trajectory,
    )

    return algo


if __name__ == '__main__':
    rllib_logger = logging.getLogger("ray.rllib")
    rllib_logger.setLevel(logging.ERROR)

    args = Config(1).get_arguments
    test_env = None
    algo = get_policy(args)

    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    if args.eval:
        test_env = HighLevelEnv(args.env_config)

    # Set up logging directory
    log_dir = os.path.join(args.log_path, 'tb_logs')
    os.makedirs(log_dir, exist_ok=True)
    algo.logdir = log_dir
    writer = SummaryWriter(log_dir=log_dir) if HAS_TB else None

    print("\n", "--- NO ERRORS FOUND, STARTING TRAINING ---")
    if HAS_TB:
        print(f"TensorBoard: run 'tensorboard --logdir {log_dir}' to view")

    time.sleep(2)
    time_acc = 0
    iters = tqdm.trange(0, args.epochs + 1, leave=True)
    os.system('clear') if os.name == 'posix' else os.system('cls')

    for i in iters:
        t = time.time()
        result = algo.train()
        time_acc += time.time() - t

        if writer is not None:
            for k, v in result.items():
                if isinstance(v, (int, float)):
                    writer.add_scalar(k, v, i)

        iters.set_description(
            f"{i}) Reward = {result.get('episode_reward_mean', 0):.2f} | "
            f"Avg. Episode Time = {round(time_acc / (i + 1), 3)} | Progress"
        )

        if i % 50 == 0:
            make_checkpoint(args, algo, i, test_env)

    if writer is not None:
        writer.close()
