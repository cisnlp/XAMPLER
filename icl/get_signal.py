import subprocess
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process SIB200")
    parser.add_argument('--model', help='Path to the pretrained model')
    parser.add_argument('--method', default='signal', help='sample selection')
    parser.add_argument('--task', default='sib200', required=True, help='sib200 or taxi')
    parser.add_argument('--data', default='eng_Latn_signal', required=True, help='data')
    parser.add_argument('--fewshot', default='1', help='CUDA visible devices')
    parser.add_argument('--gpu', default='0', help='CUDA visible devices')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    src_lang, trg_lang = args.data, args.data
    src_lang_id, trg_lang_id = 0, 0
    subprocess.run([
        'python', '../main.py',
        '--model_api_name', 'hf-causal',
        '--model_args', f'use_accelerate=True,pretrained={args.model}',
        '--src_lang', f'{src_lang}',
        '--trg_lang', f'{trg_lang}',
        '--src_lang_id', f'{src_lang_id}',
        '--trg_lang_id', f'{trg_lang_id}',
        '--task_name', f'{args.task}',
        '--task_args', f'data_dir=/path/data/{args.task}/{args.data},download_mode=load_from_disk',
        '--output_dir', '/path/outputs/',
        '--template_names', 'all_templates',
        '--method', args.method,
        '--num_fewshot', f'{args.fewshot}',
        '--batch_size', '8',
        '--device', 'cuda:0'
    ])

