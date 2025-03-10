import os
import argparse
import json
import torch
import gc
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer,T5ForConditionalGeneration
from typing import List, Union
import re
import string
import ast
from collections import Counter
import logging
import tiktoken
import time
logger = logging.getLogger(__name__)


def get_token_nums(string: str, encoding_name: str = 'o200k_base') -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        logger.info("GPU 사용 가능")
    else:
        logger.warning("GPU 사용 불가, CPU 사용")
    out_file_name = args.out_file_name
    

    com_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if com_tokenizer.pad_token is None:
        com_tokenizer.pad_token = com_tokenizer.eos_token
        com_tokenizer.pad_token_id = com_tokenizer.eos_token_id
        print("com_tokenizer.pad 없음")

    compressor = T5ForConditionalGeneration.from_pretrained(args.model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
    compressor = compressor.to(device)
    compressor.generation_config.pad_token_id = com_tokenizer.pad_token_id
    compressor.eval()


    def read_json_file(input_dir):
        lines = []

        if os.path.getsize(input_dir) == 0:
            print("파일이 비어 있습니다.")
            return lines

        with open(input_dir, "r", encoding='utf-8-sig') as file:
            try:
                try:
                    file_content = json.load(file)
                    return file_content  
                except json.JSONDecodeError:
                    file.seek(0) 
                    for line in file:
                        if line.strip():  
                            try:
                                data = json.loads(line)
                                lines.append(data)
                            except json.JSONDecodeError as e:
                                print(f"JSONDecodeError: {e} for line: {line}")
                    return lines
            except Exception as e:
                print(f"Error: {e}")
                return []


    lines = read_json_file(args.input_dir)
    eval_out = open(os.path.join(args.output_dir, out_file_name)+".json", "w")


    print("Total Samples: ", len(lines))
    
    prefix = "Instruction: Compress the imformation in the retrieved documents into a 2-sentence summary that could be used to answer the question\n"
    
    eval_prompt_temp = prefix + "Question: {question}\n{context}\nSummary: "

    padding = "max_length" if args.pad_to_max_length else False

    for idx, dic in enumerate(tqdm(lines)):
        start = time.time()
        question = dic["question"] 
        context = ''

        try:
            for idx, ctx in enumerate(dic['ctxs']):
                context += f"[Document{idx+1}]: {ctx['text']}" + "\n"
        except TypeError:
            context += f"[Document{idx+1}]:" +dic['ctxs']['text'] + "\n"
            continue
        ground_truths = dic["answers"]
        
        qa_prompt = eval_prompt_temp.format(context=context, question=question)

        qa_inputs = com_tokenizer(
            qa_prompt, return_tensors="pt", padding=padding, truncation=False
        ).to(compressor.device)
        attention_mask = qa_inputs["attention_mask"]

        summary = compressor.generate(
            input_ids=qa_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=com_tokenizer.pad_token_id
        )
        summary_decoding = com_tokenizer.decode(summary[0], skip_special_tokens=True).strip()
        for s in ['[Document1]', '[Document2]', '[Document3]', '[Document4]', '[Document5]']:
            summary_decoding = summary_decoding.replace(s,"")
        summary_decoding = summary_decoding.strip('\n')

        end = time.time()
        batch_size = int(qa_inputs.input_ids.size(0)) 
        seq_length = int(qa_inputs.input_ids.size(1)) 
        print(f"batch_size: {batch_size}, seq_length: {seq_length}")
        len_context = get_token_nums(context)
        len_summary = get_token_nums(summary_decoding)
        compress_ratio = round(len_summary/len_context,2)

        new_entry = {
            "question": question,
            "answers": ground_truths,
            "compress": summary_decoding,
            "compress_ratio": compress_ratio,
            "compress_latency": end-start,
            "seq_length": seq_length,
        }
        
        
        eval_out.write(json.dumps(new_entry, ensure_ascii=False)+"\n")
        

    eval_out.close()
    del compressor
    torch.cuda.empty_cache()
    gc.collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', type=str, help='model_name', default='./model/Qwen2.5-0.5B-pt-it-nq') # meta-llama/Meta-Llama-3-8B-Instruct 
    parser.add_argument('--input_dir', type=str, default="./recomp_data/abstractive_compressor_inputs/nq_dev_contriever_msmarco_top_5_docs.json")
    parser.add_argument('--output_dir', type=str, default="./recomp_data/evaluations/")
    parser.add_argument('--out_file_name', type=str, default="Natural_Quesiton")
    parser.add_argument('--pad_to_max_length', type=bool, default=False)
    args = parser.parse_args()
    
    main(args)