import subprocess
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process SIB200")
    parser.add_argument('--model', help='Path to the pretrained model')
    parser.add_argument('--method', default='random', required='True', help='sample selection')
    parser.add_argument('--data', default='sib200', required='True', help='sib200 or taxi')
    parser.add_argument('--fewshot', default='3', required='True', help='CUDA visible devices')
    parser.add_argument('--gpu', default='0', required='True', help='CUDA visible devices')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(f'../lang_list/{args.data}_lang_list.txt', 'r') as f:
        langs = [line.strip() for line in f.readlines()]

    for src_lang_id, src_lang in enumerate(['eng_Latn']):
        for trg_lang_id, trg_lang in enumerate(langs):
            print(src_lang, trg_lang)
            subprocess.run([
                'python', '../main.py',
                '--model_api_name', 'hf-causal',
                '--model_args', f'use_accelerate=True,pretrained={args.model}',
                '--src_lang', f'{src_lang}',
                '--trg_lang', f'{trg_lang}',
                '--src_lang_id', f'{src_lang_id}',
                '--trg_lang_id', f'{trg_lang_id}',
                '--task_name', f'{args.data}',
                '--task_args', f'data_dir=/path/data/{args.data}/{trg_lang},download_mode=load_from_disk',
                '--output_dir', '/path/outputs/',
                '--template_names', 'all_templates',
                '--method', args.method,
                '--num_fewshot', f'{args.fewshot}',
                '--batch_size', '2',
                '--device', 'cuda:0'
            ])

