import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm, trange
import json
from typing import List
import os

class IsTreeDataset(Dataset):
    def __init__(self, n_nodes, graph_type='binary', use_cot=True, use_retrieval=True, cycle_shut_down=True, n_samples=None, add_idle_tokens=0, pad = 'right'):
        self.n_nodes = n_nodes
        self.data = []
        self.tokenized_data = []
        self.labels = []
        self.graph_type = graph_type
        self.use_cot = use_cot
        self.n_vocab = 0
        self.n_samples = n_samples
        self.cycle_shut_down = cycle_shut_down
        self.add_idle_tokens = add_idle_tokens
        self.use_retrieval = use_retrieval
        self.pad = pad
        if n_samples:
            self.generate(n_samples)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_mask[idx],
                'labels': self.labels[idx]}
    
    def generate(self, n_samples):
        for i in tqdm(range(n_samples)):
            graph = self._generate_graph()
            self.data.append(graph)
        self.preprocess()
        self.prepare()
    
    def _generate_graph(self):
        if self.graph_type == 'binary':
            return self._generate_binary_graph()
        elif self.graph_type == 'random':
            return self._generate_random_graph()
        else:
            raise ValueError('wrong graph type')
    
    def _generate_binary_graph(self):
        perm = np.random.permutation(self.n_nodes)
        graph = np.zeros((self.n_nodes, self.n_nodes))
        s1 = 0
        s2 = 1
        for i in range(2, self.n_nodes):
            if np.random.rand() < 0.5:
                graph[perm[s1], perm[i]] = 1
                graph[perm[i], perm[s1]] = 1
            else:
                graph[perm[s2], perm[i]] = 1
                graph[perm[i], perm[s2]] = 1
        i = np.random.randint(2, self.n_nodes-1)
        graph[perm[i], perm[i+1]] = 1
        graph[perm[i+1], perm[i]] = 1
        return graph
    
    def _generate_random_graph(self):
        p = np.random.rand()
        graph = np.zeros((self.n_nodes, self.n_nodes))
        if p < 0.5:
            perm = np.random.permutation(self.n_nodes)
            for i in range(1, self.n_nodes):
                father = np.random.randint(0, i)
                graph[perm[father], perm[i]] = 1
                graph[perm[i], perm[father]] = 1
        else:
            another_root = np.random.randint(1, self.n_nodes)
            perm = np.random.permutation(self.n_nodes)
            for i in range(self.n_nodes):
                if i != another_root:
                    father = np.random.randint(0, self.n_nodes)
                    graph[perm[father], perm[i]] = 1
                    graph[perm[i], perm[father]] = 1
            u = np.random.randint(0, self.n_nodes)
            v = np.random.randint(0, self.n_nodes)
            while (u == v or graph[u, v] or u == another_root or v == another_root):
                u = np.random.randint(0, self.n_nodes)
                v = np.random.randint(0, self.n_nodes)
            graph[u, v] = 1
            graph[v, u] = 1
        return graph
    
    def euler_tour(self, graph, edges):
        n_nodes = graph.shape[0]
        edge_lists = [[] for _ in range(n_nodes)]
        for x, y in edges:
            edge_lists[x].append(y)
            edge_lists[y].append(x)
        visited = np.zeros(n_nodes)
        tour = []
        flag = True
        def dfs(u, fa):
            nonlocal flag
            visited[u] = 1
            tour.append(u)
            for y in edge_lists[u]:
                if not flag:
                    return
                if y == fa:
                    continue
                if self.cycle_shut_down and visited[y]:
                    flag = False
                    return
                elif not visited[y]:
                    visited[y] = 1
                    dfs(y, u)
                    if flag:
                        tour.append(u)
        dfs(0, -1)
        return tour
    def euler_tour_with_retrieval(self, graph, edges):
        n_nodes = graph.shape[0]
        stack_of_v = []
        visited = np.zeros(n_nodes)
        tour = []
        stack_of_v.append(0)
        visited[0] = 1
        cnt = 0
        start_search = "[STARTSEARCH]"
        end_search = "[ENDSEARCH]"
        father_sign = "~"
        edge_sign = "-"
        failed = "[FAILED]"
        prev_v = failed
        edge_lists = [[] for _ in range(n_nodes)]
        for x, y in edges:
            edge_lists[x].append(y)
            edge_lists[y].append(x)
        father_dict = {0: failed}
        while len(stack_of_v) > 0:
            top_v = stack_of_v[-1]
            next_v = failed
            tour.extend([top_v, start_search, top_v, father_sign, end_search]) 
            if(len(stack_of_v) > 1):
                father_v = stack_of_v[-2]
            else:
                father_v = failed
            tour.append(father_v)
            start_flag = False if prev_v != father_v else True
            if prev_v == father_v:
                tour.extend([start_search, top_v, edge_sign, end_search])
            else:
                tour.extend([start_search, top_v, edge_sign, prev_v, end_search])
            for v in edge_lists[top_v]:
                if(start_flag):
                    next_v = v
                    break
                if(v == prev_v):
                    start_flag = True
            tour.append(next_v)
            if(next_v == father_v and father_v != failed):
                tour.extend([start_search, top_v, edge_sign, next_v, end_search])
                next_v = failed
                start_flag = False
                for v in edge_lists[top_v]:
                    if(start_flag):
                        next_v = v
                        break
                    if(v == father_v):
                        start_flag = True
                tour.append(next_v)
            if(next_v == failed):
                stack_of_v.pop()
                prev_v = top_v
                continue
            else:
                tour.extend([start_search, next_v, father_sign, end_search])
                if(not visited[next_v]):
                    tour.extend([failed, next_v, father_sign, top_v])
                    father_dict[next_v] = top_v
                    stack_of_v.append(next_v)
                    visited[next_v] = 1
                    prev_v = top_v
                    continue
                else:
                    tour.append(father_dict[next_v])
                    break
        return tour                            
                    
    def istree(self, graph):
        n_nodes = graph.shape[0]
        visited = np.zeros(n_nodes)
        edge_lists = [[] for _ in range(n_nodes)]
        for i in range(n_nodes):
            for j in range(i):
                if graph[i, j]:
                    edge_lists[i].append(j)
                    edge_lists[j].append(i)
        def dfs(u):
            visited[u] = 1
            for v in edge_lists[u]:
                if not visited[v]:
                    dfs(v)
        dfs(0)
        return np.sum(visited) == n_nodes
    
    def preprocess(self):
        for graph in tqdm(self.data):
            self.tokenized_data.append(self._preprocess(graph, self.use_cot))
            self.labels.append(self.istree(graph))
        self.tokenize()

    def add_vocab(self, word):
        if word not in self.vocab:
            self.vocab[word] = self.n_vocab
            self.n_vocab += 1
    
    def add_vocab_list(self, words):
        for word in words:
            self.add_vocab(word)

    def tokenize(self):
        self.n_vocab = 0
        self.vocab = {'[IGNORE]': -100}
        self.add_vocab_list(['[', ']', ',', '-', '[PAD]', '[EOS]', '[TRUE]', '[FALSE]', ' ', '\n', '[UNK]', '[BOS]', '[STARTSEARCH]',
                             '[ENDSEARCH]','[FAILED]','~'])
        for i in range(self.n_nodes+1):
            self.add_vocab(str(i))
        self.tokenized_data = [self._tokenize(data) for data in self.tokenized_data]
        self.tokenized_labels = [int(label) for label in self.labels]
        self.vocab_inv = {v: k for k, v in self.vocab.items()}

    def _tokenize(self, data):
        if '[BOS]' in data:
            tokens = [self.vocab['[BOS]']]
            data = data.replace('[BOS]', '')
        if self.use_cot or self.use_retrieval:
            data = data.split('[EOS]')
            edges = data[0].split(',')
            for edge in edges:
                u, v = edge.split('-')
                tokens.append(self.vocab[u])
                tokens.append(self.vocab['-'])
                tokens.append(self.vocab[v])
            tokens.append(self.vocab['[EOS]'])
            tour = data[1].split(',')
            for node in tour:
                tokens.append(self.vocab[node])
            tokens.append(self.vocab['[EOS]'])
        else:
            edges = data.split('[EOS]')[0].split(',')
            for edge in edges:
                u, v = edge.split('-')
                tokens.append(self.vocab[u])
                tokens.append(self.vocab['-'])
                tokens.append(self.vocab[v])
            tokens.append(self.vocab['[EOS]'])
        for i in range(self.add_idle_tokens):
            tokens.append(self.vocab['[UNK]'])
        return tokens

    def _preprocess(self, graph, use_cot=True):
        n_nodes = graph.shape[0]
        edges = []
        for i in range(n_nodes):
            for j in range(i):
                if graph[i, j]:
                    edges.append((i, j))
        edges = np.random.permutation(edges)
        data_str = ''
        for u, v in edges:
            data_str += f'{u}-{v},'
        data_str = data_str[:-1]
        data_str += '[EOS]'
        if not self.use_retrieval and use_cot:
            tour = self.euler_tour(graph, edges)
            tour_str = ','.join([str(x) for x in tour])
            data_str += tour_str + '[EOS]'
        elif self.use_retrieval:
            tour = self.euler_tour_with_retrieval(graph, edges)
            tour_str = ','.join([str(x) for x in tour])
            data_str += tour_str + '[EOS]'
        return '[BOS]' + data_str
    
    def convert_ids_to_tokens(self, ids):
        # print(self.vocab_inv.items())
        return ' '.join([self.vocab_inv[id.item()] for id in ids])

    def prepare(self):
        for i in range(len(self.tokenized_data)):
            self.tokenized_data[i].append(self.vocab['[TRUE]'] if self.tokenized_labels[i] else self.vocab['[FALSE]'])
            self.tokenized_data[i] = torch.tensor(self.tokenized_data[i])
        pad_len = max([len(data) for data in self.tokenized_data])
        # left pad with [PAD]
        self.attention_mask = []
        self.labels = []
        if(self.pad == 'left'):
            for i in tqdm(range(len(self.tokenized_data))):
                self.attention_mask.append(torch.cat((torch.tensor([0] * (pad_len - len(self.tokenized_data[i]))), torch.tensor([1] * len(self.tokenized_data[i])))))
                self.tokenized_data[i] = torch.cat((torch.tensor([self.vocab['[PAD]']] * (pad_len - len(self.tokenized_data[i]))), self.tokenized_data[i]))
                pos_of_first_eos = torch.where(self.tokenized_data[i] == self.vocab['[EOS]'])[0][0]
                self.labels.append(torch.cat((torch.tensor([-100] * (pos_of_first_eos + 1)), self.tokenized_data[i][pos_of_first_eos+1:])))
                pos_of_end_search = torch.where(self.tokenized_data[i] == self.vocab['[ENDSEARCH]'])[0] + 1
                self.labels[i][pos_of_end_search] = -100
                pos_of_unk_tokens = torch.where(self.tokenized_data[i] == self.vocab['[UNK]'])[0]
                self.labels[i][pos_of_unk_tokens] = -100
        else:
            # right pad
            for i in tqdm(range(len(self.tokenized_data))):
                self.attention_mask.append(torch.cat((torch.tensor([1] * len(self.tokenized_data[i])), torch.tensor([0] * (pad_len - len(self.tokenized_data[i]))))))
                padding_position = len(self.tokenized_data[i])
                self.tokenized_data[i] = torch.cat((self.tokenized_data[i], torch.tensor([self.vocab['[PAD]']] * (pad_len - len(self.tokenized_data[i])))))
                pos_of_first_eos = torch.where(self.tokenized_data[i] == self.vocab['[EOS]'])[0][0]
                self.labels.append(torch.cat((torch.tensor([-100] * (pos_of_first_eos + 1)), 
                                              self.tokenized_data[i][pos_of_first_eos+1:padding_position], 
                                              torch.tensor([-100] * (pad_len - padding_position)))))
                pos_of_end_search = torch.where(self.tokenized_data[i] == self.vocab['[ENDSEARCH]'])[0] + 1
                self.labels[i][pos_of_end_search] = -100
                pos_of_unk_tokens = torch.where(self.tokenized_data[i] == self.vocab['[UNK]'])[0]
                self.labels[i][pos_of_unk_tokens] = -100            
        self.input_ids = torch.stack(self.tokenized_data)
        self.input_ids = self.input_ids.to(torch.long)
        self.attention_mask = torch.stack(self.attention_mask)
        self.attention_mask = self.attention_mask.to(torch.long)
        self.labels = torch.stack(self.labels)
        self.labels = self.labels.to(torch.long)

    def save(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save({
            'input_ids': self.input_ids,
            'attention_mask': self.attention_mask,
            'labels': self.labels
        }, f'{output_dir}/tensor_data.pt')
        metadata = {
            'vocab': self.vocab,
            'vocab_inv': self.vocab_inv,
            'n_vocab': self.n_vocab,
            'use_cot': self.use_cot,
            'graph_type': self.graph_type,
            'n_nodes': self.n_nodes,
            'n_samples': self.n_samples,
        }
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load(self, input_dir):
        with open(f'{input_dir}/metadata.json', 'r') as f:
            metadata = json.load(f)
        self.vocab = metadata['vocab']
        self.vocab_inv = {int(k): v for k, v in metadata['vocab_inv'].items()}
        self.n_vocab = metadata['n_vocab']
        self.use_cot = metadata['use_cot']
        self.graph_type = metadata['graph_type']
        self.n_nodes = metadata['n_nodes']
        self.n_samples = metadata['n_samples']
        tensor_data = torch.load(f'{input_dir}/tensor_data.pt')
        self.input_ids = tensor_data['input_ids'].to(torch.long)
        self.attention_mask = tensor_data['attention_mask'].to(torch.long)
        self.labels = tensor_data['labels'].to(torch.long)
        assert (self.input_ids.shape[0] == self.attention_mask.shape[0] == self.labels.shape[0] == self.n_samples)
        self.tokenized_data = [self.input_ids[i] for i in range(self.n_samples)]
        self.tokenized_labels = [(self.input_ids[i][-1] == self.vocab['[TRUE]']) for i in range(self.n_samples)]

def load_dataset(input_dir):
    dataset = IsTreeDataset(1, 'binary', True, False, None)
    dataset.load(input_dir)
    return dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_nodes', type=int, default=16)
    parser.add_argument('--graph_type', type=str, default='random')
    parser.add_argument('--task_type', type=str, default = 'no_cot')
    parser.add_argument('--size', type=int, default = 500000)
    parser.add_argument('--pad', type=str, default = 'right')
    args = parser.parse_args()
    PATH = "./data"
    if(args.task_type == 'no_cot'):
        train_dataset = IsTreeDataset(args.n_nodes, args.graph_type, False, False, False, args.size, 0, pad = args.pad)
        val_dataset = IsTreeDataset(args.n_nodes, args.graph_type, False, False, False, 5000, 0, pad = args.pad)
    elif(args.task_type == 'cot'):
        train_dataset = IsTreeDataset(args.n_nodes, args.graph_type, True, False, True, args.size, 0, pad = args.pad)
        val_dataset = IsTreeDataset(args.n_nodes, args.graph_type, True, False, True, 5000, 0, pad = args.pad)
    elif(args.task_type == 'retrieval'):
        train_dataset = IsTreeDataset(args.n_nodes, args.graph_type, False, True, False, args.size, 0, pad = args.pad)
        val_dataset = IsTreeDataset(args.n_nodes, args.graph_type, True, True, True, 5000, 0, pad = args.pad)
    if(args.pad == 'left'):
        name = args.task_type + '_' + args.graph_type + '_' + str(args.n_nodes) + '_' + str(args.size)
    else:
        name = args.task_type + '_' + args.graph_type + '_' + str(args.n_nodes) + '_' + str(args.size) + '_' + args.pad
    train_dataset.save(f'{PATH}/rnn/{name}/')
    val_dataset.save(f'{PATH}/rnn/{name}/val')
