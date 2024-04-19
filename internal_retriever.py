import argparse
import logging
import math
import os
import pickle
import sys
from collections import Counter
from os import listdir
from os.path import isfile, join
import gc
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, Wav2Vec2FeatureExtractor,
                          Wav2Vec2Model)
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
from datasets import load_from_disk

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='mala', type=str, required=True, help='model name')
    parser.add_argument('--task', type=str, required=True, help='sib200')
    parser.add_argument('--device', default='0', type=str, required=False, help='GPU ID')
    args = parser.parse_args()

    langs = {}
    with open(f'lang_list/{args.task}_lang_list.txt', 'r') as f:
        lines = f.readlines()
        langs = [line.strip() for line in lines]
    device = torch.device('cuda:' + args.device)

    if args.model == 'sbert':
        embed_loader = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    else:
        embed_loader = SentenceTransformer(f'save/{args.model}')
    embed_loader = embed_loader.to(device)
    embed_loader.eval()

    # the training set for all languages are the same, i.e., eng
    dataset = load_from_disk(f'data/{args.task}/{langs[0]}')
    hrl_sentences = [data['sent'] for data in dataset['train']]
    hrl_labels = [data['label'] for data in dataset['train']]
    hrl_emb = embed_loader.encode(hrl_sentences, convert_to_tensor=True)
    for lang in langs:
        dataset = load_from_disk(f'data/{args.task}/{lang}')
        lrl_sentences = [data['sent'] for data in dataset['test']]
        lrl_labels = [data['label'] for data in dataset['test']]
        lrl_emb = embed_loader.encode(lrl_sentences, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(lrl_emb, hrl_emb)
        top_results = torch.topk(cos_scores, k=100)[1]
        result_d = {}
        for sentence, label, top_result in zip(lrl_sentences, lrl_labels, top_results):
            top_result = [result.item() for result in top_result]
            top_label = [hrl_labels[i] for i in top_result]
            result_d[sentence] = top_result
        with open(f'save/{lang}_{args.model}_{args.task}.pickle', 'wb') as handle:
            pickle.dump(result_d, handle, protocol=pickle.HIGHEST_PROTOCOL)
    torch.cuda.empty_cache()

