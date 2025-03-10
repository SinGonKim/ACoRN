import json
from transformers import pipeline
from tqdm import tqdm

import re
import random
random.seed(0)
from collections import defaultdict


def replace_numbers_with_random_and_track(string, min_value=1, max_value=1000):
    numbers_found = False

    def replace_with_random(match):
        nonlocal numbers_found
        numbers_found = True  
        return str(random.randint(min_value, max_value)) 

    result = re.sub(r'\b\d+\b', replace_with_random, string)
    return result, numbers_found

with open('./mydata/popQA/base/test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
def pick_random_list(list1, list2):
    import random
    random.seed(0)
    return random.choice([list1, list2])

fill_mask = pipeline("fill-mask", model='roberta-large')



total = 0
var_info = defaultdict(int)
factual_error = 0
cnt = 0
number = 0 
blank = 0 
title = 0 



new_data = []
for idx, entry in enumerate(tqdm(data), start=1):
    check = 0
    entry['evidentiality_idx'] = []
    entry['evidentiality_cnt'] = 0
    for idx, ctx in enumerate(entry['ctxs']):
        for ans in entry['answers']:
            if ans.lower() in ctx['text'].lower():
                check += 1
                entry['evidentiality_cnt'] += 1
                entry['evidentiality_idx'].append(idx)
                break
    if check <2:
        var_info[check] += 1
    else:
        factual_error_idx = random.choice([-1] + entry['evidentiality_idx'])
        if factual_error_idx == -1:
            var_info[check] += 1
            new_data.append(entry)
            total += 1
            continue
        factual_error_ctx = entry['ctxs'][factual_error_idx]

        answers = []
        for answer in entry['answers']: 
            if answer.strip().lower() in factual_error_ctx['text'].strip().lower():
                answers.append(answer.strip())
        import re
        # Replace all answers in ctx['text'] with <mask>
        temp = factual_error_ctx['text']
        for answer in answers:
            # Use regex to ensure exact matches and handle overlapping or partial replacements
            factual_error_ctx['text'] = re.sub(re.escape(answer.lower()), "<mask>", factual_error_ctx['text'].lower(), flags=re.IGNORECASE)
        masked_text = factual_error_ctx['text']
            
        # Use ROBERTA to predict the masked answer
        question = "Question: " + entry['question']
        predictions = fill_mask(question + "\n" + masked_text)
        for idx, prediction in enumerate(predictions):
            try:
                if str(prediction['token_str']).strip() not in [str(a).strip() for a in answers]:
                    factual_error_ctx['text'] = factual_error_ctx['text'].replace("<mask>", str(prediction['token_str']).strip())
                    cnt += 1
                    factual_error_ctx['use'] = 'roberta'
                    factual_error_ctx['prediction'] = "factual_error"
                    factual_error_ctx['has_answer'] = False
                    entry['evidentiality_cnt']-= 1
                    entry['evidentiality_idx'].remove(factual_error_idx)
                    var_info[check-1] += 1
                    factual_error += 1
                    break
            except TypeError:
                continue
        else:
                min_random = 0 
                max_random = 999  
                modified_text, modified = replace_numbers_with_random_and_track(answer.strip(), min_random, max_random)
                if modified:
                    factual_error_ctx['text'] = factual_error_ctx['text'].replace("<mask>", modified_text)
                    number += 1
                    factual_error_ctx['use'] = 'number'
                    factual_error_ctx['prediction'] = "factual_error"
                    factual_error_ctx['has_answer'] = False
                    entry['evidentiality_cnt']-= 1
                    entry['evidentiality_idx'].remove(factual_error_idx)
                    var_info[check-1] += 1
                    factual_error += 1
                elif all(answer not in factual_error_ctx['title'].strip() for answer in answers):
                    factual_error_ctx['text'] = factual_error_ctx['text'].replace("<mask>", factual_error_ctx['title'])
                    title += 1
                    factual_error_ctx['use'] = 'title'
                    factual_error_ctx['prediction'] = "factual_error"
                    factual_error_ctx['has_answer'] = False
                    entry['evidentiality_cnt']-= 1
                    entry['evidentiality_idx'].remove(factual_error_idx)
                    var_info[check-1] += 1
                    factual_error += 1
                else:
                    factual_error_ctx['text'] = temp
                    var_info[check] += 1
    new_data.append(entry)
    total += 1

# Save the modified data to C.json
with open('./mydata/popQA/base/test_aug.json', 'w', encoding='utf-8') as file:
    for item in new_data:
        json.dump(item, file, ensure_ascii=False)
        file.write('\n')
log_file = open('./mydata/popQA/base/test_aug.txt', "w")
log_file.write(f"데이터 총 {total}개, zero: {var_info[0]}개, one: {var_info[1]}개, two: {var_info[2]}개, three: {var_info[3]}개, four: {var_info[4]}개, five: {var_info[5]}개, factual_error: {factual_error}개, roberta-large: {cnt}개, number: {number}개, title: {title} blank: {blank}개")
log_file.close()
