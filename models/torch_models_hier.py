"""
Pure PyTorch model for HHMARL high-level Commander policy.
Replaces RLlib's RecurrentNetwork dependency.
Maintains compatible forward() interface for self-play inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from algorithms.ppo import SlimFC, add_time_dimension


N_OPP_HL = 2  # change for sensing
OBS_OPP = 10
OBS_DIM = 14 + OBS_OPP * N_OPP_HL


class CommanderGru(nn.Module):
    """High-level Commander policy with GRU recurrence."""

    def __init__(self):
        super().__init__()
        self.num_outputs = N_OPP_HL + 1  # Discrete action space
        self._shared_layer = SlimFC(500, 500, activation_fn=nn.Tanh,
                                     initializer=torch.nn.init.orthogonal_)
        self.rnn_act = nn.GRU(200, 200, batch_first=True)
        self.rnn_val = nn.GRU(200, 200, batch_first=True)

        self.inp1 = SlimFC(4, 50, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(N_OPP_HL * OBS_OPP, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(10, 50, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp4 = SlimFC(OBS_DIM, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.act_out = SlimFC(500, self.num_outputs, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self.v1 = SlimFC(OBS_DIM + 1, 100, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v2 = SlimFC(OBS_DIM + 1, 100, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v3 = SlimFC(OBS_DIM + 1, 100, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v4 = SlimFC(3 * (OBS_DIM + 1), 200, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(500, 1, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self._val = None
        self._inp1 = None
        self._inp2 = None
        self._inp3 = None
        self._inp4 = None
        self._v1 = None
        self._v2 = None
        self._v3 = None
        self._v4 = None

    @property
    def shared_layer(self):
        return self._shared_layer

    @shared_layer.setter
    def shared_layer(self, layer):
        self._shared_layer = layer

    def get_initial_state(self):
        return [torch.zeros(200), torch.zeros(200)]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        self._inp1 = obs["obs_1_own"][:, :4]
        self._inp2 = obs["obs_1_own"][:, 4:4 + N_OPP_HL * OBS_OPP]
        self._inp3 = obs["obs_1_own"][:, 4 + N_OPP_HL * OBS_OPP:]
        self._inp4 = obs["obs_1_own"]

        self._v1 = torch.cat((obs["obs_1_own"], obs["act_1_own"]), dim=1)
        self._v2 = torch.cat((obs["obs_2"], obs["act_2"]), dim=1)
        self._v3 = torch.cat((obs["obs_3"], obs["act_3"]), dim=1)
        self._v4 = torch.cat((self._v1, self._v2, self._v3), dim=1)

        output, new_state = self.forward_rnn(input_dict, state, seq_lens)
        output = torch.reshape(output, [-1, self.num_outputs])
        return output, new_state

    def forward_rnn(self, input_dict, state, seq_lens):
        x = torch.cat((self.inp1(self._inp1), self.inp2(self._inp2),
                       self.inp3(self._inp3)), dim=1)
        x_full = self.inp4(self._inp4)

        if state is not None and len(state) > 0 and state[0] is not None:
            h0 = torch.unsqueeze(state[0], 0)
        else:
            h0 = torch.zeros(1, x_full.shape[0], 200, device=x_full.device)

        y, h = self.rnn_act(
            add_time_dimension(x_full, seq_lens=seq_lens),
            h0)
        x_full = F.normalize(x_full + y.reshape(-1, 200))
        x = torch.cat((x, x_full), dim=1)
        x = self._shared_layer(x)
        x = self.act_out(x)

        z = torch.cat((self.v1(self._v1), self.v2(self._v2),
                       self.v3(self._v3)), dim=1)
        z_full = self.v4(self._v4)

        if state is not None and len(state) > 1 and state[1] is not None:
            k0 = torch.unsqueeze(state[1], 0)
        else:
            k0 = torch.zeros(1, z_full.shape[0], 200, device=z_full.device)

        w, k = self.rnn_val(
            add_time_dimension(z_full, seq_lens=seq_lens),
            k0)
        z_full = F.normalize(z_full + w.reshape(-1, 200))
        z = torch.cat((z, z_full), dim=1)
        z = self._shared_layer(z)
        self._val = self.val_out(z)

        return x, [torch.squeeze(h, 0), torch.squeeze(k, 0)]

    def value_function(self):
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])
