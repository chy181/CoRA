import torch
from torch import nn

import sys
sys.path.insert(0,"ts_benchmark/baselines/LLM/submodules/GPT4TS")

from ts_benchmark.baselines.LLM.submodules.GPT4TS import GPT4TS

class GPT4TSModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        # config.pred_len = config.horizon
        self.model = GPT4TS.GPT4TS(config, device)
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device):        
        output = self.model(x_enc, device)
        return output

    def get_settings(self):
        return (
            self.model.in_layer.out_features,
            self.model.patch_size,
            self.model.stride,
            self.model.patch_num,
        )

    def forcast_for_plugin(self, inputs, x_mark_enc, dec_inp, x_mark_dec, device=None):
        return self.model.forcast_for_plugin(inputs)

    def denorm_for_plugin(self, inputs):
        return self.model.denorm_for_plugin(inputs)

    def patchify_for_plugin(self, inputs):
        return self.model.patchify_for_plugin(inputs)
