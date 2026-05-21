"""
Main file for training low-level heterogeneous agents.
Uses custom PPO implementation instead of Ray RLlib.

HETEROGENEOUS: Agent IDs to AC types: 1->1, 2->2, 3->1, 4->2
"""

import os
import time
import shutil
import tqdm
import torch
import numpy as np
from pathlib import Path
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False

from algorithms.ppo import MultiAgentPPO, PPOPolicy, SlimFC
from models.torch_models_hetero import Fight1, Fight2, Esc1, Esc2
from config import Config
from envs.env_hetero import LowLevelEnv

ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3
OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29
POLICY_DIR = 'policies'


def create_shared_layer():
    """Create the shared layer used by all models."""
    return SlimFC(500, 500, activation_fn=torch.nn.Tanh,
                  initializer=torch.nn.init.orthogonal_)


def update_logs(args, results_dir, level, epoch):
    """Save checkpoints to experiment log directory."""
    result_dir = os.path.join(args.log_path, 'checkpoint')
    try:
        shutil.rmtree(result_dir)
    except Exception:
        pass
    if os.path.exists(results_dir):
        shutil.copytree(results_dir, result_dir, symlinks=False, dirs_exist_ok=False)


def evaluate(args, algo, env, epoch, level, it):
    """Evaluations are stored as pictures of combat scenarios."""

    def cc_obs(obs, agent_id):
        if agent_id == 1:
            return {
                "obs_1_own": obs[1],
                "obs_2": obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            }
        elif agent_id == 2:
            return {
                "obs_1_own": obs[2],
                "obs_2": obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }

    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    while not done:
        actions = {}
        for ag_id in state.keys():
            a = algo.compute_single_action(
                observation=cc_obs(state, ag_id),
                state=torch.zeros(1),
                policy_id=f"ac{ag_id}_policy",
                explore=False)
            actions[ag_id] = a[0]

        state, rew, term, trunc, _ = env.step(actions)
        done = term["__all__"] or trunc["__all__"]
        for r in rew.values():
            reward += r

        step += 1
        if args.render:
            try:
                env.plot(Path(args.log_path, "current.png"))
            except RuntimeError:
                pass
            time.sleep(0.18)

    reward = round(reward, 3)
    try:
        env.plot(Path(args.log_path,
                      f"Ep_{epoch}_It_{step}_Lv{level}_Rew_{reward}.png"))
    except RuntimeError:
        pass


def make_checkpoint(args, algo, log_dir, epoch, level, env=None):
    algo.save(os.path.join(args.log_path, 'checkpoint'))
    update_logs(args, os.path.join(args.log_path, 'checkpoint'), level, epoch)
    for it in range(2):
        if args.level >= 3:
            algo.export_policy_model(os.path.join(os.path.dirname(__file__), POLICY_DIR),
                                     f'ac{it + 1}_policy')
            policy_name = f'L{args.level}_AC{it + 1}_{args.agent_mode}'
            old_path = os.path.join(POLICY_DIR, 'model.pt')
            new_path = os.path.join(POLICY_DIR, f'{policy_name}.pt')
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
        if args.eval and epoch % 500 == 0:
            evaluate(args, algo, env, epoch, level, it)


def get_policy(args):
    """
    Create MultiAgentPPO with heterogeneous policies.
    Agents get assigned Fight1/Fight2 or Esc1/Esc2 networks.
    """

    # Centralized critic observation function
    def central_critic_observer(agent_obs):
        """Augment observations for centralized critic."""
        new_obs = {
            1: {
                "obs_1_own": agent_obs[1],
                "obs_2": agent_obs[2],
                "act_1_own": np.zeros(ACTION_DIM_AC1),
                "act_2": np.zeros(ACTION_DIM_AC2),
            },
            2: {
                "obs_1_own": agent_obs[2],
                "obs_2": agent_obs[1],
                "act_1_own": np.zeros(ACTION_DIM_AC2),
                "act_2": np.zeros(ACTION_DIM_AC1),
            }
        }
        return new_obs

    def postprocess_trajectory(episode_data):
        """
        Replace zero actions in augmented observations with actual actions.
        Normalizes heading (/12) and speed (/8) to [0,1] range, matching
        the original RLlib CustomCallback.on_postprocess_trajectory.
        """
        policy_actions = {}
        for pid, traj in episode_data.items():
            actions_list = [t["action"].cpu().numpy() for t in traj]
            policy_actions[pid] = np.stack(actions_list, axis=0)

        agent_acts = {}
        if "ac1_policy" in policy_actions:
            agent_acts[1] = policy_actions["ac1_policy"]
        if "ac2_policy" in policy_actions:
            agent_acts[2] = policy_actions["ac2_policy"]

        def norm_own(act):
            a = act.astype(np.float32).copy()
            a[0] /= 12.0
            a[1] /= 8.0
            return a

        for pid, traj in episode_data.items():
            agent_id = 1 if pid == "ac1_policy" else 2
            other_id = 2 if agent_id == 1 else 1

            for t_idx, transition in enumerate(traj):
                obs = transition["aug_obs"]

                # Own actions: fill all dims with normalized values
                own = norm_own(agent_acts[agent_id][t_idx])
                obs["act_1_own"] = own

                # Other agent's actions: overwrite first 3 elements of existing array
                # (preserves original array shape — matches RLlib callback behavior)
                if other_id in agent_acts and t_idx < len(agent_acts[other_id]):
                    other = agent_acts[other_id][t_idx].astype(np.float32).copy()
                    other[0] /= 12.0
                    other[1] /= 8.0
                    n = min(len(other), 3)
                    obs["act_2"][:n] = other[:n]

    # Create shared layer
    shared_layer = create_shared_layer()

    # Create models based on agent_mode
    if args.agent_mode == "escape":
        model1 = Esc1()
        model2 = Esc2()
        action_dims1 = [13, 9, 2, 2]
        action_dims2 = [13, 9, 2]
    else:
        model1 = Fight1()
        model2 = Fight2()
        action_dims1 = [13, 9, 2, 2]
        action_dims2 = [13, 9, 2]

    # Share the shared layer across models (same as original RLlib behavior)
    model1.shared_layer = shared_layer
    model2.shared_layer = shared_layer

    # Create policies
    policies = {
        "ac1_policy": PPOPolicy(
            model=model1,
            action_dims=action_dims1,
            lr=1e-4,
            clip_param=0.25,
            entropy_coeff=0.01,
            vf_coeff=0.5,
            max_grad_norm=0.5,
        ),
        "ac2_policy": PPOPolicy(
            model=model2,
            action_dims=action_dims2,
            lr=1e-4,
            clip_param=0.25,
            entropy_coeff=0.01,
            vf_coeff=0.5,
            max_grad_norm=0.5,
        ),
    }

    # Create env factory
    def env_fn():
        return LowLevelEnv(args.env_config)

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
    args = Config(0).get_arguments
    test_env = None
    algo = get_policy(args)

    if args.restore:
        if args.restore_path:
            algo.restore(args.restore_path)
        else:
            algo.restore(os.path.join(args.log_path, "checkpoint"))
    if args.eval:
        test_env = LowLevelEnv(args.env_config)

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
            f"Level = {args.level} | "
            f"Avg. Episode Time = {round(time_acc / (i + 1), 3)} | Progress"
        )

        if i % 50 == 0:
            make_checkpoint(args, algo, log_dir, i, args.level, test_env)

    if writer is not None:
        writer.close()
