from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP 
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.nn as nn
import numpy as np
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
import wandb
import os
import json

from rnn.utils import set_seed, parse_args
from rnn.data import load_dataset, IsTreeDataset
from rnn.model import get_transformer_model, get_rnn_model, get_hybrid_model

import torch.distributed as dist
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

def main():
    args = parse_args()
    # Setup for distributed computing
    args.device=int(os.environ['RANK'])
    args.local_rank=int(os.environ['RANK'])
    args.rank=int(os.environ['RANK'])
    if args.report_to_wandb:
        wandb.init()
    set_seed(args.seed)
    train_dataset = load_dataset(args.dataset_dir)
    val_dataset = load_dataset(os.path.join(args.dataset_dir, 'val'))
    if args.model_type == 'transformer':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_transformer_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                max_position_embeddings=config['max_position_embeddings']
            )
        else:
            model = get_transformer_model(
                train_dataset,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=20,
                num_attention_heads=8,
                max_position_embeddings=4096
            )
    elif args.model_type == 'rnn':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_rnn_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers']
            )
        else:
            model = get_rnn_model(
                train_dataset,
                hidden_size = 128,
                num_hidden_layers= 10
            )
    elif args.model_type == 'hybrid':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_hybrid_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                max_position_embeddings=config['max_position_embeddings'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size']
            )
        else:
            model = get_hybrid_model(
                train_dataset,
                hidden_size=128,
                num_hidden_layers=9
            )
    else:
        raise NotImplementedError
    import transformers
    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,          # output directory
        num_train_epochs=args.total_training_samples // len(train_dataset),              # total number of training epochs
        per_device_train_batch_size=args.batch_size,  # batch size per device during training
        per_device_eval_batch_size=args.batch_size,   # batch size for evaluation
        warmup_steps=0,                # number of warmup steps for learning rate scheduler
        weight_decay=0,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=2000,
        evaluation_strategy="steps",     # Evaluation is done (and logged) every logging_steps
        learning_rate = args.lr
    )
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=-1)
        predictions = predictions[:, -labels.shape[1]:-1]
        labels = labels[:, 1:]
        exact_match_cnt = 0
        cnt = 0
        for prediction, label in zip(predictions, labels):
            correct = (prediction == label) + (label == -100)
            cnt += 1
            exact_match_cnt += correct.all()
        return {"exact_match": exact_match_cnt / cnt}
    trainer = transformers.Trainer(
        model=model, args=training_args, train_dataset=train_dataset,
        compute_metrics=compute_metrics,
        eval_dataset = val_dataset
    )

    # Train the model
    # resume_checkpoint_path = "./data/retrieval_binary_128_100000_transformer_2m_run_parallel/checkpoint-25000"
    # trainer.train(resume_checkpoint_path)
    
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)

if __name__ == '__main__':
    main()