import os
import argparse
import json
import torch
import gc
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM 
from huggingface_hub import login
from typing import List, Union
import re
import string
import ast
from collections import Counter
import logging
import time
logger = logging.getLogger(__name__)
def parse_prediction(pred):

    stop_idx = -1
    for word in ["Question:", "\n\n", "</s>"]:
        idx = pred.find(word)
        if stop_idx == -1 or (idx != -1 and idx < stop_idx):
            stop_idx = idx
    if stop_idx != -1:
        pred = pred[:stop_idx].strip()
    return pred
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def em_score(prediction: str, ground_truths: Union[str, List[str]]):
    correct = np.max([int(normalize_answer(prediction) == normalize_answer(gt)) for gt in ground_truths])
    return correct

def f1_score(prediction: str, ground_truths: Union[str, List[str]]):
    final_metric = {'f1': 0, 'precision': 0, 'recall': 0, 'acc': 0}
    for ground_truth in ground_truths:
        normalized_prediction = normalize_answer(prediction)
        normalized_ground_truth = normalize_answer(ground_truth)
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        acc = 1.0 if normalized_ground_truth in normalized_prediction else 0.0
        for k in ['f1', 'precision', 'recall', 'acc']:
            final_metric[k] = max(eval(k), final_metric[k])
    return final_metric['f1'], final_metric['precision'], final_metric['recall'], final_metric['acc']


def generate_answer(qa_prompt, tokenizer, model):

    system_prompt = "You are a helpful AI assistant! Refer to the summary and answer the question using individual words instead of full sentences."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": qa_prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_dict = True,
        return_tensors="pt"
    ).to(model.device)
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=32,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.0
    )
    generated_tokens = outputs[:, inputs.input_ids.shape[-1]:]
    pred = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    pred = pred.strip()
    return pred
def append_json_line(filepath, data):
    with open(filepath, 'a') as file:
        json.dump(data, file)
        file.write('\n') 

def add_results(filepath, original_question, output, ground_truth, pred, summary, em, f1, pre, recall, acc):
    new_entry = {
        "question": original_question,
        "passages": output,
        "answers": ground_truth,
        "prediction": pred,
        "compress": summary,
        "em": em,
        "f1": f1,
        "precision": pre,
        "recall": recall,
        "accuracy": acc
    }
    
    append_json_line(filepath, new_entry)  

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU 사용 가능")
    else:
        device = torch.device("cpu")
        logger.warning("GPU 사용 불가, CPU 사용")
    out_file_name = args.out_file_name
    
    inf_tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    if inf_tokenizer.pad_token is None:
        inf_tokenizer.pad_token = inf_tokenizer.eos_token
        inf_tokenizer.pad_token_id = inf_tokenizer.eos_token_id
    
    elif inf_tokenizer.eos_token is None:
        inf_tokenizer.eos_token = inf_tokenizer.pad_token
        inf_tokenizer.eos_token_id = inf_tokenizer.pad_token_id


    inf_model = AutoModelForCausalLM.from_pretrained(args.target_model_name, device_map="auto", load_in_8bit=False, torch_dtype=torch.float16)
    inf_model.generation_config.pad_token_id = inf_tokenizer.pad_token_id
    inf_model.eval()
    


    with open(args.input_dir, "r") as fin:
        list_data_dict = []
    
        for line in fin:
            if line.strip():
                try:
                    data = json.loads(line)
                    list_data_dict.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} for line: {line}")
    eval_out = open(os.path.join(args.output_dir, out_file_name+".json"), "w")
    log_file = open(os.path.join(args.output_dir, out_file_name+".txt"), "w")

        
    total_em = 0
    total_f1 = 0
    total_pre = 0
    total_recall = 0
    total_acc = 0
    total_compress_ratio = 0
    total_time = 0
    inference_time = 0
    total_key = 0
    print("Total Samples: ", len(list_data_dict))

    eval_answer_prompt_temp = "Summary: {summary}\n Question: {question}\n Answer: "
    zero_shot_prompt_temp = "Question: {question}\n Answer: "


    for dic in tqdm(list_data_dict):
        start = time.time()
        question = dic["question"]     
        ground_truths = dic["answers"]
        
        summary_decoding = dic["compress"]
        if dic["compress"] == "This passage doesn't contain relevant information to the question.":
            inf_prompt = zero_shot_prompt_temp.format(question=question)
        else:
            inf_prompt = eval_answer_prompt_temp.format(summary=summary_decoding, question=question)

        pred = generate_answer(inf_prompt, inf_tokenizer, inf_model) 
        pred = parse_prediction(pred)
        end = time.time()
        
        em = em_score(pred, ground_truths)
        f1, pre, recall, acc = f1_score(pred, ground_truths)

        evidentiality = 0
        for answer in ground_truths:
            if answer in summary_decoding:
                evidentiality = 1
                break
        new_entry = {
            "question": question,
            "answers": ground_truths,
            "prediction": pred,
            "compress": summary_decoding,
            "em": str(em),
            "f1": str(f1),
            "compress_ratio": dic["compress_ratio"],
            "inference_latency": end-start,
            "total_latency": end-start+dic["compress_latency"],
            "precision": str(pre),
            "recall": str(recall),
            "accuracy": str(acc),
            "evidentiality": str(evidentiality),
        }
        
        
        eval_out.write(json.dumps(new_entry, ensure_ascii=False)+"\n")
        
        total_em += em
        total_f1 += f1
        total_pre += pre
        total_recall += recall
        total_acc += acc
        total_compress_ratio += dic["compress_ratio"]
        total_time += end-start+dic["compress_latency"]
        inference_time += end-start
        total_key += evidentiality
        # flops += dic['flops']
        
    total_em /= len(list_data_dict)
    total_f1 /= len(list_data_dict)
    total_pre /= len(list_data_dict)
    total_recall /= len(list_data_dict)
    total_acc /= len(list_data_dict)
    total_compress_ratio /= len(list_data_dict)
    total_time /= len(list_data_dict)
    inference_time /= len(list_data_dict)
    total_key /= len(list_data_dict)
    print("EM: ", total_em)
    print("F1: ", total_f1)
    print("Precision: ", total_pre)
    print("Recall: ", total_recall)
    print("Accuracy: ", total_acc)
    print("comprerss_ratio: ", total_compress_ratio)
    print("total_time: ", total_time)
    print("inference_time: ", inference_time)
    print("total_key: ", total_key)
    log_file.write(f"EM: {total_em}\n")
    log_file.write(f"F1: {total_f1}\n")
    log_file.write(f"Precision: {total_pre}\n")
    log_file.write(f"Recall: {total_recall}\n")
    log_file.write(f"Accuracy: {total_acc}\n")
    log_file.write(f"comprerss_ratio: {total_compress_ratio}\n")
    log_file.write(f"total_time: {total_time}\n")
    log_file.write(f"inference_time: {inference_time}\n")
    log_file.write(f"total_key: {total_key}\n")
    eval_out.close()
    log_file.close()
    del inf_model
    torch.cuda.empty_cache()
    gc.collect()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_model_name', '-tm', type=str, help='target_model_name', default='meta-llama/Meta-Llama-3-8B-Instruct') # mistralai/Mistral-7B-Instruct-v0.3
    parser.add_argument('--input_dir', type=str, default="./recomp_data/abstractive_compressor_inputs/nq_dev_contriever_msmarco_top_5_docs.json")
    parser.add_argument('--output_dir', type=str, default="./recomp_data/evaluations/")
    parser.add_argument('--dataset_name', '--dataset', type=str, default="nq")
    parser.add_argument('--out_file_name', type=str, default="Natural_Quesiton")
    args = parser.parse_args()
    main(args)