"""
Evaluation script for HHMARL.
Evaluates trained Commander (high-level) or low-level policies.
Uses custom models instead of RLlib.
"""

import numpy as np
import torch
import os
import tqdm
from config import Config
import time
import json
from pathlib import Path
from envs.env_hier import HighLevelEnv
from models.torch_models_hier import CommanderGru

N_OPP_HL = 2  # sensing
OBS_DIM = 14 + 10 * N_OPP_HL

MODEL_NAME = "Commander_3_vs_3"  # name of commander model in folder 'results'
N_EVALS = 1000


class SimplePolicy:
    """Lightweight wrapper for a trained model to provide compute_single_action."""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def compute_single_action(self, obs, state, explore=False):
        with torch.no_grad():
            obs_t = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    obs_t[k] = torch.from_numpy(v).unsqueeze(0).float()
                else:
                    obs_t[k] = v

            if state is not None:
                if not isinstance(state, list):
                    state = [state]
                state_t = [s.unsqueeze(0) if s.dim() == 1 else s for s in state]
            else:
                state_t = None

            logits, new_state = self.model(
                {"obs": obs_t},
                state=state_t,
                seq_lens=torch.tensor([1]))

            # Commander has Discrete action space — take argmax
            action = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy()

        return action, [s.cpu() for s in new_state] if new_state else [], {}


def cc_obs(obs):
    return {
        "obs_1_own": obs,
        "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
        "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
        "act_1_own": np.zeros(1),
        "act_2": np.zeros(1),
        "act_3": np.zeros(1),
    }


def evaluate(args, env, algo, epoch, eval_stats, eval_log):
    state, _ = env.reset()
    reward = 0
    done = False
    step = 0
    info = {}

    while not done:
        actions = {}
        states = [torch.zeros(200), torch.zeros(200)]

        if args.eval_hl:
            for ag_id, ag_s in state.items():
                a, new_states, _ = algo.compute_single_action(
                    obs=cc_obs(ag_s), state=states, explore=False)
                actions[ag_id] = a
                if new_states and len(new_states) >= 2:
                    states[0] = new_states[0]
                    states[1] = new_states[1]
        else:
            # If no commander, assign closest opponent for each agent.
            for n in range(1, args.num_agents + 1):
                actions[n] = 1

        state, rew, hist, trunc, info = env.step(actions)
        for r in rew.values():
            reward += r
        done = hist["__all__"] or trunc["__all__"]
        step += 1

        for k, v in info.items():
            eval_stats[k] += v
        eval_stats["total_n_actions"] += 1

    if epoch % 100 == 0:
        env.plot(Path(eval_log,
                      f"Ep_{epoch}_Step_{step}_Rew_{round(reward, 2)}.png"))


def postprocess_eval(ev, eval_file):
    """Calculate fractions and save results."""
    win = (ev["agents_win"] / N_EVALS) * 100
    lose = (ev["opps_win"] / N_EVALS) * 100
    draw = (ev["draw"] / N_EVALS) * 100
    fight = (ev["agent_fight"] / max(ev["agent_steps"], 1)) * 100
    esc = (ev["agent_escape"] / max(ev["agent_steps"], 1)) * 100
    fight_opp = (ev["opp_fight"] / max(ev["opp_steps"], 1)) * 100
    esc_opp = (ev["opp_escape"] / max(ev["opp_steps"], 1)) * 100
    opp1 = (ev["opp1"] / max(ev["agent_fight"], 1)) * 100
    opp2 = (ev["opp2"] / max(ev["agent_fight"], 1)) * 100
    opp3 = (ev["opp3"] / max(ev["agent_fight"], 1)) * 100
    evals = {
        "win": win, "lose": lose, "draw": draw,
        "fight": fight, "esc": esc,
        "fight_opp": fight_opp, "esc_opp": esc_opp,
        "opp1": opp1, "opp2": opp2, "opp3": opp3,
    }
    for k, v in evals.items():
        print(f"{k}: {round(v, 2)}")
    with open(eval_file, 'w') as file:
        json.dump(evals, file, indent=3)


if __name__ == "__main__":
    t1 = time.time()
    args = Config(2).get_arguments

    log_base = os.path.join(os.getcwd(), 'results')
    check = os.path.join(log_base, MODEL_NAME, 'checkpoint')
    config = "Commander_" if args.eval_hl else "Low-Level_"
    config = config + f"{args.num_agents}-vs-{args.num_opps}"
    eval_log = os.path.join(log_base, "EVAL_" + config)
    eval_file = os.path.join(eval_log, f"Metrics_{config}.json")
    if not os.path.exists(eval_log):
        os.makedirs(eval_log)

    env = HighLevelEnv(args.env_config)

    # Load commander policy if evaluating with high-level
    if args.eval_hl:
        model = CommanderGru()
        policy_path = os.path.join(check, 'commander_policy.pt')
        if os.path.exists(policy_path):
            checkpoint = torch.load(policy_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
        policy = SimplePolicy(model)
    else:
        policy = None

    eval_stats = {
        "agents_win": 0, "opps_win": 0, "draw": 0,
        "agent_fight": 0, "agent_escape": 0,
        "opp_fight": 0, "opp_escape": 0,
        "agent_steps": 0, "opp_steps": 0, "total_n_actions": 0,
        "opp1": 0, "opp2": 0, "opp3": 0,
    }

    iters = tqdm.trange(0, N_EVALS, leave=True)
    for n in iters:
        evaluate(args, env, policy, n, eval_stats, eval_log)

    print("------RESULTS:")
    postprocess_eval(eval_stats, eval_file)
    print(f"------TIME: {round(time.time() - t1, 3)} sec.")
