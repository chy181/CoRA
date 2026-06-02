import sys
sys.path.insert(0, "ts_benchmark/baselines/LLM/submodules/UniTime")

from einops import rearrange
import torch
from torch import nn
import json
from pathlib import Path

from ts_benchmark.baselines.LLM.submodules.UniTime import unitime

class UniTimeModel(nn.Module):
    def __init__(
        self,
        config,
        device
    ):
        super().__init__()
        self.device = device
        self.mask_rate = config.mask_rate

        self.config = config
        self.model = unitime.UniTime(config)

        self.data_id = config.dataset + '_' + str(config.seq_len) + '_' + str(config.pred_len)
        

        if Path(config.instruct_path).exists():
            with open(config.instruct_path, 'r') as f:
                instruct_list = json.load(f)
                self.instruct = instruct_list.get(config.dataset, "")
        else:
            self.instruct = ""

        self.info = [self.data_id, config.seq_len, config.stride, self.instruct]
       
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, device): 

        b, t, n = x_enc.shape
        mask = torch.rand((b, t, n)).to(self.device)
        mask[mask < self.mask_rate] = 0  # masked
        mask[mask >= self.mask_rate] = 1  # remained
        inp = x_enc.masked_fill(mask == 0, 0)
        output = self.model(self.info, inp, mask)
        return output

    def _full_mask(self, inputs):
        return torch.ones_like(inputs, device=inputs.device)

    def _patch_num(self):
        seq_len = self.config.seq_len
        patch_len = self.config.patch_len
        stride = self.config.stride
        if seq_len <= patch_len:
            padded_len = patch_len
        elif seq_len % stride == 0:
            padded_len = seq_len
        else:
            padded_len = (seq_len // stride) * stride + patch_len
        return (padded_len - patch_len) // stride + 1

    def get_settings(self):
        return (
            self.model.d_model,
            self.config.patch_len,
            self.config.stride,
            self._patch_num(),
        )

    def forcast_for_plugin(self, inputs, x_mark_enc, dec_inp, x_mark_dec, device=None):
        return self.model.forcast_for_plugin(self.info, inputs, self._full_mask(inputs))

    def denorm_for_plugin(self, inputs):
        return self.model.denorm_for_plugin(inputs)

    def patchify_for_plugin(self, inputs):
        return self.model.patchify_for_plugin(
            inputs,
            self.config.seq_len,
            self.config.stride,
        )
