import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path
try:
    import huggingface_hub.errors as hf_errors
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    if not hasattr(hf_errors, "EntryNotFoundError"):
        hf_errors.EntryNotFoundError = EntryNotFoundError
    if not hasattr(hf_errors, "LocalEntryNotFoundError"):
        hf_errors.LocalEntryNotFoundError = LocalEntryNotFoundError
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from ..models.GPT2_arch import AccustumGPT2Model

class Encoder_PCA(nn.Module):
    def __init__(self, input_dim, word_embedding, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_PCA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        
        self.word_embedding = word_embedding.T

    def forward(self, x):
        B = x.shape[0]
        if self.word_embedding.ndim == 2:
            self.word_embedding = self.word_embedding.repeat(B, 1, 1)
        elif self.word_embedding.shape[0] != B:
            self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)

        x = self.linear(x)

        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)

        x_time = x

        q = x.transpose(0, 1)
        k = v = self.word_embedding.transpose(0, 1)
        x, _ = self.cross_attention(q, k, v)

        x = x.transpose(0, 1)

        return x_time, x

class Model(nn.Module):
    def __init__(self, configs, device):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
    
        self.task_name = configs.task_name

        gpt2_path = Path("ts_benchmark/baselines/LLM/checkpoints/gpt2")
        if getattr(configs, "pretrain", 1) and gpt2_path.exists():
            self.gpt2 = AccustumGPT2Model.from_pretrained(str(gpt2_path), output_attentions=True, output_hidden_states=True)
            self.gpt2_text = AccustumGPT2Model.from_pretrained(str(gpt2_path), output_attentions=True, output_hidden_states=True)
        elif getattr(configs, "pretrain", 1):
            self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.gpt2_text = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        else:
            print("------------------no pretrain------------------")
            gpt_config = GPT2Config(
                n_embd=configs.d_model,
                n_layer=configs.gpt_layers,
                n_head=getattr(configs, "n_heads", 12),
                output_attentions=True,
                output_hidden_states=True,
            )
            self.gpt2 = AccustumGPT2Model(gpt_config)
            self.gpt2_text = AccustumGPT2Model(gpt_config)

        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        self.gpt2_text.h = self.gpt2_text.h[:configs.gpt_layers]

        if get_peft_model is not None:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=configs.r,
                lora_alpha=configs.lora_alpha,
                lora_dropout=configs.lora_dropout,
                target_modules=["c_attn"],
            )
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
        elif getattr(configs, "pretrain", 1):
            raise ImportError("CALF LoRA requires a compatible peft installation.")

        if Path(configs.word_embedding_path).exists():
            word_embedding = torch.tensor(
                torch.load(configs.word_embedding_path, weights_only=False)
            ).to(device=device)
        else:
            word_embedding = torch.randn(configs.d_model, 500, device=device)
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name or 'lora' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for i, (name, param) in enumerate(self.gpt2_text.named_parameters()):
            if 'wpe' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.time_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])
        
        self.text_proj = nn.ModuleList([nn.Linear(configs.d_model, configs.d_model, bias=False) for _ in range(configs.gpt_layers+1)])

        self.in_layer = Encoder_PCA(configs.seq_len, word_embedding, hidden_dim=configs.d_model)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.out_layer = nn.Linear(configs.d_model, configs.pred_len)
        elif self.task_name == 'classification':
            self.out_layer = nn.Linear(configs.d_model * configs.enc_in, configs.num_class)
        elif self.task_name == 'imputation':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)
        elif self.task_name == 'anomaly_detection':
            self.out_layer = nn.Linear(configs.d_model, configs.seq_len)

        for layer in (self.gpt2_text, self.gpt2, self.in_layer, self.out_layer, self.time_proj, self.text_proj):
            layer.to(device=device)
            layer.train()
        
        self.cnt = 0

    def _encode_forecast(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        return {
            "outputs_time": outputs_time,
            "outputs_text": outputs_text,
            "intermidiate_time": intermidiate_feat_time,
            "intermidiate_text": intermidiate_feat_text,
            "means": means,
            "stdev": stdev,
        }

    def _decode_forecast(self, outputs_time, outputs_text, means, stdev, M):
        outputs_time = self.out_layer(outputs_time[:, -M:, :])
        outputs_text = self.out_layer(outputs_text[:, -M:, :])

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
        }

    def forecast(self, x):
        encoded = self._encode_forecast(x)
        output = self._decode_forecast(
            encoded["outputs_time"],
            encoded["outputs_text"],
            encoded["means"],
            encoded["stdev"],
            x.shape[-1],
        )
        output["intermidiate_time"] = encoded["intermidiate_time"]
        output["intermidiate_text"] = encoded["intermidiate_text"]
        return output

    def forcast_for_plugin(self, x):
        encoded = self._encode_forecast(x)
        self.plugin_means = encoded["means"]
        self.plugin_stdev = encoded["stdev"]
        output = self._decode_forecast(
            encoded["outputs_time"],
            encoded["outputs_text"],
            encoded["means"],
            encoded["stdev"],
            x.shape[-1],
        )
        embedding = encoded["outputs_time"][:, -x.shape[-1]:, :].unsqueeze(2)
        return output["outputs_time"], embedding

    def denorm_for_plugin(self, x):
        return x * self.plugin_stdev + self.plugin_means

    def patchify_for_plugin(self, x):
        return rearrange(x, 'b l c -> b c 1 l')


    def classification(self, x):
        B, L, M = x.shape

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)
        
        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])
        
        outputs_time = outputs_time.reshape(B, -1)
        outputs_text = outputs_text.reshape(B, -1)
        
        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)
        
        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }
    

    def imputation(self, x, mask):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        x = x.masked_fill(mask == 0, 0)

        stdev = torch.sqrt(torch.sum(x**2, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5).unsqueeze(1).detach()
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }

    def anomaly_detection(self, x):
        B, L, M = x.shape

        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach() 
        x /= stdev

        x = rearrange(x, 'b l m -> b m l')

        outputs_time1, outputs_text1 = self.in_layer(x)

        outputs_time, intermidiate_feat_time = self.gpt2(inputs_embeds=outputs_time1)
        outputs_text, intermidiate_feat_text = self.gpt2_text(inputs_embeds=outputs_text1)
        # residue connection
        outputs_time += outputs_time1
        outputs_text += outputs_text1
        
        intermidiate_feat_time = tuple([self.time_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_time))])
        intermidiate_feat_text = tuple([self.text_proj[idx](feat) for idx, feat in enumerate(list(intermidiate_feat_text))])

        outputs_time = self.out_layer(outputs_time)
        outputs_text = self.out_layer(outputs_text)

        outputs_time = rearrange(outputs_time, 'b m l -> b l m')
        outputs_text = rearrange(outputs_text, 'b m l -> b l m')

        outputs_text = outputs_text * stdev + means
        outputs_time = outputs_time * stdev + means

        return {
            'outputs_text': outputs_text,
            'outputs_time':outputs_time,
            'intermidiate_time':intermidiate_feat_time,
            'intermidiate_text':intermidiate_feat_text,
        }


    def forward(self, x, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            output = self.forecast(x)
        if self.task_name == 'classification':
            output = self.classification(x)
        if self.task_name == "imputation":
            output = self.imputation(x, mask)
        if self.task_name == "anomaly_detection":
            output = self.anomaly_detection(x)
        return output
