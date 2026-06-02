# import numpy as np
import torch
import torch.nn as nn
# from torch import optim

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from transformers import BertTokenizer, BertModel
from einops import rearrange
# from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

class GPT4TS(nn.Module):
    
    def __init__(self, configs, device):
        super(GPT4TS, self).__init__()
        self.is_gpt = configs.is_gpt
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1
        
        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = GPT2Model.from_pretrained('ts_benchmark/baselines/LLM/checkpoints/gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = GPT2Model(GPT2Config())
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            # print("gpt2 = {}".format(self.gpt2))
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        for layer in (self.gpt2, self.in_layer, self.out_layer):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def _encode_patches(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')

        outputs = self.in_layer(x)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        return outputs, means, stdev, B, M

    def _decode_patches(self, outputs, means, stdev, B, M):
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means

        return outputs

    def forward(self, x, itr):
        outputs, means, stdev, B, M = self._encode_patches(x)
        return self._decode_patches(outputs, means, stdev, B, M)

    def forcast_for_plugin(self, x):
        outputs, means, stdev, B, M = self._encode_patches(x)
        self.plugin_means = means
        self.plugin_stdev = stdev
        forecast = self._decode_patches(outputs, means, stdev, B, M)
        embedding = rearrange(outputs, '(b m) n d -> b m n d', b=B, m=M)
        return forecast, embedding

    def denorm_for_plugin(self, x):
        return x * self.plugin_stdev + self.plugin_means

    def patchify_for_plugin(self, x):
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        return x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
