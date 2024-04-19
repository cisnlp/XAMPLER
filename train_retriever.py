import os
from os import listdir
from os.path import isfile, join
import argparse
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, models
import json
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from datasets import load_from_disk
import random

random.seed(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='sib200', type=str, help='sib200 or taxi')
    parser.add_argument("--data", default='eng_Latn_signal', type=str)
    parser.add_argument("--base_model", default='sbert', type=str)
    parser.add_argument('--tune', action='store_true', help='tuned model')
    parser.add_argument('--device', default='0', type=str, required=False, help='GPU ID')
    args = parser.parse_args()

    # Define the model. Either from scratch of by loading a pre-trained model
    device = "cuda:" + args.device if torch.cuda.is_available() else "cpu"
    if args.base_model == 'sbert':
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device=device)
    elif args.base_model == 'glot500':
        word_embedding_model = models.Transformer("cis-lmu/glot500-base", max_seq_length=256, model_args={'device': device})
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device=device)
    else:
        model = SentenceTransformer(f'save/{args.base_model}', device=device)

    train_examples = []
    sentences1, sentences2, scores = [], [], []
    cnt = 1
    
    infile = os.path.join(f'data/{args.task}/{args.data}')
    dataset = load_from_disk(infile)
    src_sents = [data['sent'] for data in dataset['train']]
    trg_sents = [data['sent'] for data in dataset['test']]
    src_labels = [data['label'] for data in dataset['train']]
    trg_labels = [data['label'] for data in dataset['test']]
    
    if args.tune:
        outputs = [f for f in os.listdir('outputs') if f.startswith('example') and f'{args.task}' in f and f'{args.data}.' in f and 'fewshot=1.' in f and 'save' in f]
    else:
        outputs = [f for f in os.listdir('outputs') if f.startswith('example') and f'{args.task}' in f and f'{args.data}.' in f and 'fewshot=1.' in f and 'llama' in f]
    assert len(outputs) == 1, outputs
    for output in outputs:
        print(output)
        with open(f'outputs/{output}') as f:
            lines = f.readlines()[-len(src_sents):]
            for line in lines:
                d = json.loads(line)
                src_sent = src_sents[d['fewshot_idx'][0]]
                trg_sent = trg_sents[d['fewshot_idx'][0]]
                src_label = src_labels[d['fewshot_idx'][0]]
                trg_label = trg_labels[d['fewshot_idx'][0]]
                pred = d['pred']
                target = d['target']
                assert src_sent in d['ctx']
                assert trg_sent in d['ctx']
                assert trg_label == target

                # train
                if random.random() < 0.9:
                    if pred == target:
                        train_examples.append(InputExample(texts=[src_sent, trg_sent], label=1.0))
                        cnt += 1
                    else:
                        train_examples.append(InputExample(texts=[src_sent, trg_sent], label=0.0))
                # dev
                else:
                    sentences1.append(src_sent)
                    sentences2.append(trg_sent)
                    if pred == target:
                        scores.append(1.0)
                    else:
                        scores.append(0.0)
    print(cnt, len(train_examples), cnt / len(train_examples))
    evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

    # Define your train dataset, the dataloader and the train loss
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.ContrastiveLoss(model)
    # train_loss = losses.CosineSimilarityLoss(model)

    # Tune the model
    if args.tune:
        output_path = f'save/{args.base_model}_{args.task}_{args.data}_tune'
    else:
        output_path = f'save/{args.base_model}_{args.task}_{args.data}'
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=50,
        optimizer_params={'lr': 2e-05},
        evaluator=evaluator,
        # output_path=f'save/{args.base_model}_{args.task}_{args.data}',
        output_path=output_path,
    )

