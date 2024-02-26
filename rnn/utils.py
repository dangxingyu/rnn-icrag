import argparse
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='transformer', choices=['transformer', 'rnn', 'hybrid'])
    parser.add_argument('--dataset_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_training_samples', type=int, default=200000)
    parser.add_argument('--log_interval', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--warmup_samples', type=int, default=10000)
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--report_to_wandb', action='store_true')
    parser.add_argument('--model_config_path', type=str, default=None)
    
    return parser.parse_args()