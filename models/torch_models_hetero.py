"""
Pure PyTorch models for HHMARL low-level heterogeneous agents.
Replaces RLlib's TorchModelV2 / RecurrentNetwork dependency.
Maintains compatible forward() interface for self-play inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Local imports
from algorithms.ppo import SlimFC, add_time_dimension


ACTION_DIM_AC1 = 4
ACTION_DIM_AC2 = 3

OBS_AC1 = 26
OBS_AC2 = 24
OBS_ESC_AC1 = 30
OBS_ESC_AC2 = 29

SS_AGENT_AC1 = 12
SS_AGENT_AC2 = 10


class BaseModel(nn.Module):
    """Base class providing get_initial_state for non-recurrent models."""

    def get_initial_state(self):
        return []


class Esc1(BaseModel):
    """Escape policy for AC1 (type 1 aircraft)."""

    def __init__(self):
        super().__init__()
        self.num_outputs = 13 + 9 + 2 + 2  # MultiDiscrete [13,9,2,2]
        self._shared_layer = SlimFC(500, 500, activation_fn=nn.Tanh,
                                     initializer=torch.nn.init.orthogonal_)

        self.inp1 = SlimFC(7, 150, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(18, 250, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(5, 100, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.act_out = SlimFC(500, self.num_outputs, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)
        self.inp1_val = SlimFC(OBS_ESC_AC1 + ACTION_DIM_AC1 + OBS_ESC_AC2 + ACTION_DIM_AC2,
                               500, activation_fn=nn.Tanh,
                               initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(500, 1, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self._v1 = None

    @property
    def shared_layer(self):
        return self._shared_layer

    @shared_layer.setter
    def shared_layer(self, layer):
        self._shared_layer = layer

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _inp1 = obs["obs_1_own"][:, :7]
        _inp2 = obs["obs_1_own"][:, 7:25]
        _inp3 = obs["obs_1_own"][:, 25:]
        self._v1 = torch.cat((obs["obs_1_own"], obs["act_1_own"],
                               obs["obs_2"], obs["act_2"]), dim=1)

        x = torch.cat((self.inp1(_inp1), self.inp2(_inp2), self.inp3(_inp3)), dim=1)
        x = self._shared_layer(x)
        x = self.act_out(x)
        return x, []

    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        x = self.inp1_val(self._v1)
        x = self._shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])


class Esc2(BaseModel):
    """Escape policy for AC2 (type 2 aircraft)."""

    def __init__(self):
        super().__init__()
        self.num_outputs = 13 + 9 + 2  # MultiDiscrete [13,9,2]
        self._shared_layer = SlimFC(500, 500, activation_fn=nn.Tanh,
                                     initializer=torch.nn.init.orthogonal_)

        self.inp1 = SlimFC(6, 150, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(18, 250, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(5, 100, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.act_out = SlimFC(500, self.num_outputs, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)
        self.inp1_val = SlimFC(OBS_ESC_AC1 + ACTION_DIM_AC1 + OBS_ESC_AC2 + ACTION_DIM_AC2,
                               500, activation_fn=nn.Tanh,
                               initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(500, 1, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self._v1 = None

    @property
    def shared_layer(self):
        return self._shared_layer

    @shared_layer.setter
    def shared_layer(self, layer):
        self._shared_layer = layer

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _inp1 = obs["obs_1_own"][:, :6]
        _inp2 = obs["obs_1_own"][:, 6:24]
        _inp3 = obs["obs_1_own"][:, 24:]
        self._v1 = torch.cat((obs["obs_1_own"], obs["act_1_own"],
                               obs["obs_2"], obs["act_2"]), dim=1)

        x = torch.cat((self.inp1(_inp1), self.inp2(_inp2), self.inp3(_inp3)), dim=1)
        x = self._shared_layer(x)
        x = self.act_out(x)
        return x, []

    def value_function(self):
        assert self._v1 is not None, "must call forward first!"
        x = self.inp1_val(self._v1)
        x = self._shared_layer(x)
        x = self.val_out(x)
        return torch.reshape(x, [-1])


class Fight1(BaseModel):
    """Fight policy for AC1 (type 1 aircraft). Uses attention over sequence."""

    def __init__(self):
        super().__init__()
        self.num_outputs = 13 + 9 + 2 + 2  # MultiDiscrete [13,9,2,2]
        self._shared_layer = SlimFC(500, 500, activation_fn=nn.Tanh,
                                     initializer=torch.nn.init.orthogonal_)
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        self.inp1 = SlimFC(SS_AGENT_AC1, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OBS_AC1 - SS_AGENT_AC1, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(OBS_AC1, 100, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.act_out = SlimFC(500, self.num_outputs, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)
        self.v1 = SlimFC(OBS_AC1 + ACTION_DIM_AC1, 175, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v2 = SlimFC(OBS_AC2 + ACTION_DIM_AC2, 175, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v3 = SlimFC(OBS_AC1 + ACTION_DIM_AC1 + OBS_AC2 + ACTION_DIM_AC2,
                         150, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(500, 1, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self._val = None
        self._v1 = None
        self._v2 = None
        self._v3 = None

    @property
    def shared_layer(self):
        return self._shared_layer

    @shared_layer.setter
    def shared_layer(self, layer):
        self._shared_layer = layer

    def get_initial_state(self):
        return [torch.zeros(1)]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _inp1 = obs["obs_1_own"][:, :SS_AGENT_AC1]
        _inp2 = obs["obs_1_own"][:, SS_AGENT_AC1:]
        _inp3 = obs["obs_1_own"]
        batch_len = _inp1.shape[0]

        self._v1 = torch.cat((obs["obs_1_own"], obs["act_1_own"]), dim=1)
        self._v2 = torch.cat((obs["obs_2"], obs["act_2"]), dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        x = torch.cat((self.inp1(_inp1), self.inp2(_inp2)), dim=1)
        x_full = self.inp3(_inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = F.normalize(x_full + x_att.reshape((batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self._shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._v1), self.v2(self._v2)), dim=1)
        y_full = self.v3(self._v3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = F.normalize(y_full + y_att.reshape((batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self._shared_layer(y)
        self._val = self.val_out(y)

        return x, []

    def value_function(self):
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])


class Fight2(BaseModel):
    """Fight policy for AC2 (type 2 aircraft). Uses attention over sequence."""

    def __init__(self):
        super().__init__()
        self.num_outputs = 13 + 9 + 2  # MultiDiscrete [13,9,2]
        self._shared_layer = SlimFC(500, 500, activation_fn=nn.Tanh,
                                     initializer=torch.nn.init.orthogonal_)
        self.att_act = nn.MultiheadAttention(100, 2, batch_first=True)
        self.att_val = nn.MultiheadAttention(150, 2, batch_first=True)

        self.inp1 = SlimFC(SS_AGENT_AC2, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp2 = SlimFC(OBS_AC2 - SS_AGENT_AC2, 200, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.inp3 = SlimFC(OBS_AC2, 100, activation_fn=nn.Tanh,
                           initializer=torch.nn.init.orthogonal_)
        self.act_out = SlimFC(500, self.num_outputs, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)
        self.v1 = SlimFC(OBS_AC2 + ACTION_DIM_AC2, 175, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v2 = SlimFC(OBS_AC1 + ACTION_DIM_AC1, 175, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.v3 = SlimFC(OBS_AC1 + ACTION_DIM_AC1 + OBS_AC2 + ACTION_DIM_AC2,
                         150, activation_fn=nn.Tanh,
                         initializer=torch.nn.init.orthogonal_)
        self.val_out = SlimFC(500, 1, activation_fn=None,
                              initializer=torch.nn.init.orthogonal_)

        self._val = None
        self._v1 = None
        self._v2 = None
        self._v3 = None

    @property
    def shared_layer(self):
        return self._shared_layer

    @shared_layer.setter
    def shared_layer(self, layer):
        self._shared_layer = layer

    def get_initial_state(self):
        return [torch.zeros(1)]

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        _inp1 = obs["obs_1_own"][:, :SS_AGENT_AC2]
        _inp2 = obs["obs_1_own"][:, SS_AGENT_AC2:]
        _inp3 = obs["obs_1_own"]
        batch_len = _inp1.shape[0]

        self._v1 = torch.cat((obs["obs_1_own"], obs["act_1_own"]), dim=1)
        self._v2 = torch.cat((obs["obs_2"], obs["act_2"]), dim=1)
        self._v3 = torch.cat((self._v1, self._v2), dim=1)

        x = torch.cat((self.inp1(_inp1), self.inp2(_inp2)), dim=1)
        x_full = self.inp3(_inp3)
        x_ft = add_time_dimension(x_full, seq_lens=seq_lens)
        x_att, _ = self.att_act(x_ft, x_ft, x_ft, need_weights=False)
        x_full = F.normalize(x_full + x_att.reshape((batch_len, -1)))

        x = torch.cat((x, x_full), dim=1)
        x = self._shared_layer(x)
        x = self.act_out(x)

        y = torch.cat((self.v1(self._v1), self.v2(self._v2)), dim=1)
        y_full = self.v3(self._v3)
        y_ft = add_time_dimension(y_full, seq_lens=seq_lens)
        y_att, _ = self.att_val(y_ft, y_ft, y_ft, need_weights=False)
        y_full = F.normalize(y_full + y_att.reshape((batch_len, -1)))

        y = torch.cat((y, y_full), dim=1)
        y = self._shared_layer(y)
        self._val = self.val_out(y)

        return x, []

    def value_function(self):
        assert self._val is not None, "must call forward first!"
        return torch.reshape(self._val, [-1])
