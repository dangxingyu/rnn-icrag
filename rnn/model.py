from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP
import torch 
import torch.nn as nn
import numpy as np
from rnn.mamba import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from rnn.hybrid import HybridMambaLMHeadModel


def get_transformer_config(
        dataset,
        hidden_size=768,
        intermediate_size=1024,
        num_hidden_layers=12,
        num_attention_heads=12,
        max_position_embeddings=4096,
):
    config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_attention_heads
    config.vocab_size = dataset.n_vocab
    config.max_position_embeddings = max_position_embeddings
    config.pad_token_id = dataset.vocab['[PAD]']
    config.bos_token_id = dataset.vocab['[BOS]']
    config.eos_token_id = dataset.vocab['[EOS]']
    return config


def get_transformer_model(
        dataset,
        hidden_size=32,
        intermediate_size=128,
        num_hidden_layers=16,
        num_attention_heads=4,
        max_position_embeddings=4096
    ):
    config = get_transformer_config(
        dataset,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings
    )
    model = AutoModelForCausalLM.from_config(config)
    embed_matrix = torch.zeros_like(model.model.embed_tokens.weight)
    for i in range(model.config.vocab_size):
        for j in range(model.config.hidden_size):
            if j % 2 == 0:
                embed_matrix[i][j] = np.sin(i / (10000 ** (j / model.config.hidden_size)))
            else:
                embed_matrix[i][j] = np.cos(i / (10000 ** ((j - 1) / model.config.hidden_size)))
    print(embed_matrix)
    model.model.embed_tokens = nn.Embedding.from_pretrained(embed_matrix)
    model.model.embed_tokens.weight.requires_grad = False
    model.lm_head.weight = model.model.embed_tokens.weight
    model.lm_head.weight.requires_grad = False
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    return model

def get_rnn_config(
        dataset,
        hidden_size=768,
        num_hidden_layers=12,
        ):
    ssm_config = MambaConfig(
        d_model=hidden_size,
        n_layer=num_hidden_layers,
        vocab_size=dataset.n_vocab,
    )
    return ssm_config

def  get_rnn_model(
        dataset,
        hidden_size=768,
        num_hidden_layers=12,
        ):
    config = get_rnn_config(
        dataset,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
    )
    model = MambaLMHeadModel(config).to("cuda")
    embed_matrix = torch.zeros_like(model.backbone.embedding.weight)
    print(embed_matrix.shape)
    for i in range(embed_matrix.shape[0]):
        for j in range(embed_matrix.shape[1]):
            if j % 2 == 0:
                embed_matrix[i][j] = np.sin(i / (10000 ** (j / embed_matrix.shape[1])))
            else:
                embed_matrix[i][j] = np.cos(i / (10000 ** ((j - 1) / embed_matrix.shape[1])))
    model.backbone.embedding = nn.Embedding.from_pretrained(embed_matrix)
    model.backbone.embedding.weight.requires_grad = False
    model.lm_head.weight = model.backbone.embedding.weight
    model.lm_head.weight.requires_grad = False
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    return model

def get_hybrid_config(
        dataset,
        hidden_size=768,
        num_hidden_layers=12,
        max_position_embeddings = 4096,
        num_attention_heads = 4,
        intermediate_size = 1024
        ):
    ssm_config = MambaConfig(
        d_model=hidden_size,
        n_layer=num_hidden_layers,
        vocab_size=dataset.n_vocab,
    )
    config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
    config.hidden_size = hidden_size
    config.intermediate_size = intermediate_size
    config.num_hidden_layers = num_hidden_layers
    config.num_attention_heads = num_attention_heads
    config.num_key_value_heads = num_attention_heads
    config.vocab_size = dataset.n_vocab
    config.max_position_embeddings = max_position_embeddings
    config.pad_token_id = dataset.vocab['[PAD]']
    config.bos_token_id = dataset.vocab['[BOS]']
    config.eos_token_id = dataset.vocab['[EOS]']
    ssm_config.llama_cfg = config
    return ssm_config



def get_hybrid_model(
    dataset,
    hidden_size=768,
    num_hidden_layers=12,
    max_position_embeddings = 4096,
    num_attention_heads = 4,
    intermediate_size = 1024
):
    config = get_hybrid_config(
        dataset,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        max_position_embeddings = max_position_embeddings,
        num_attention_heads = num_attention_heads,
        intermediate_size = intermediate_size
    )
    model = HybridMambaLMHeadModel(config).to("cuda")
    embed_matrix = torch.zeros_like(model.backbone.embedding.weight)
    print(embed_matrix.shape)
    for i in range(embed_matrix.shape[0]):
        for j in range(embed_matrix.shape[1]):
            if j % 2 == 0:
                embed_matrix[i][j] = np.sin(i / (10000 ** (j / embed_matrix.shape[1])))
            else:
                embed_matrix[i][j] = np.cos(i / (10000 ** ((j - 1) / embed_matrix.shape[1])))
    model.backbone.embedding = nn.Embedding.from_pretrained(embed_matrix)
    model.backbone.embedding.weight.requires_grad = False
    model.lm_head.weight = model.backbone.embedding.weight
    model.lm_head.weight.requires_grad = False
    print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    return model


if __name__ == '__main__':
    pass