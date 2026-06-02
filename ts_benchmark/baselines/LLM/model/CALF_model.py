import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/CALF")

from ts_benchmark.baselines.LLM.submodules.CALF.models import CALF

class CALFModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        # config.pred_len = config.horizon
        self.model = CALF.Model(config, device)

       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc)
        return output['outputs_time']

    def get_settings(self):
        return (
            self.model.out_layer.in_features,
            self.model.in_layer.linear.in_features,
            self.model.in_layer.linear.in_features,
            1,
        )

    def forcast_for_plugin(self, inputs, x_mark_enc, dec_inp, x_mark_dec, device=None):
        return self.model.forcast_for_plugin(inputs)

    def denorm_for_plugin(self, inputs):
        return self.model.denorm_for_plugin(inputs)

    def patchify_for_plugin(self, inputs):
        return self.model.patchify_for_plugin(inputs)
