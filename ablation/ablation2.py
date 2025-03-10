## Showing Evidential performance
import os
import argparse
import json
import torch
import gc
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM 
from typing import List, Union
import re
import string
import ast
from collections import Counter
import logging
import time
logger = logging.getLogger(__name__)
def parse_prediction(pred):
    # truncate after to the answer
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
    # if "-it" in args.target_model_name:
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

# JSON 파일에 한 줄씩 추가하는 함수
def append_json_line(filepath, data):
    with open(filepath, 'a') as file:
        json.dump(data, file)
        file.write('\n')  # 개별 entry가 한 줄씩 저장되도록 개행 추가

# 결과 데이터를 추가하는 함수
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
    
    append_json_line(filepath, new_entry)  # 한 줄씩 추가하여 저장

def main(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU available!")
    else:
        device = torch.device("cpu")
        logger.warning("GPU unavailable, Use CPU")
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
    eval_out = open(os.path.join(args.output_dir, out_file_name)+".json", "w")
    log_file = open(os.path.join(args.output_dir, out_file_name+".txt"), "w")

        
    total_em_e = 0
    total_f1_e = 0
    total_em_a = 0
    total_f1_a = 0
    print("Total Samples: ", len(list_data_dict))
    eval_answer_prompt_temp = "{Documents}\n Question: {question}\n Answer: "
    zero_shot_prompt_temp = "Question: {question}\n Answer: "

    for dic in tqdm(list_data_dict):
        question = dic["question"] 
        ground_truths = dic["answers"]
        
        Document = dic['summary_e']
        if Document == "":
            inf_prompt = zero_shot_prompt_temp.format(question=question)
        else:
            inf_prompt = eval_answer_prompt_temp.format(Documents=Document, question=question)
        
        pred_e = generate_answer(inf_prompt, inf_tokenizer, inf_model) 
        pred_e = parse_prediction(pred_e)
        
        em_e = em_score(pred_e, ground_truths)
        f1_e, pre, recall, acc = f1_score(pred_e, ground_truths)
        Document = dic['summary_a']
        inf_prompt = eval_answer_prompt_temp.format(Documents=Document, question=question) 
        
        pred_a = generate_answer(inf_prompt, inf_tokenizer, inf_model) 
        pred_a = parse_prediction(pred_a)
        
        em_a = em_score(pred_a, ground_truths)
        f1_a, pre, recall, acc = f1_score(pred_a, ground_truths)
        
        new_entry = {
            "question": question,
            "answers": ground_truths,
            "prediction_e": pred_e,
            "em_e": str(em_e),
            "f1_e": str(f1_e),
            "prediction_a": pred_a,
            "em_a": str(em_a),
            "f1_a": str(f1_a),
        }
        
        
        eval_out.write(json.dumps(new_entry, ensure_ascii=False)+"\n")
        
        total_em_e += em_e
        total_f1_e += f1_e
        total_em_a += em_a
        total_f1_a += f1_a

        # flops += dic['flops']
        
    total_em_e /= len(list_data_dict)
    total_f1_e /= len(list_data_dict)
    total_em_a /= len(list_data_dict)
    total_f1_a /= len(list_data_dict)
    # flops /= len(list_data_dict)
    print("EMe: ", total_em_e)
    print("F1e: ", total_f1_e)
    print("EMa: ", total_em_a)
    print("F1a: ", total_f1_a)
    log_file.write(f"EMe: {total_em_e}\n")
    log_file.write(f"F1e: {total_f1_e}\n")
    log_file.write(f"EMa: {total_em_a}\n")
    log_file.write(f"F1a: {total_f1_a}\n")
    eval_out.close()
    log_file.close()
    del inf_model
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