import json
from transformers import pipeline
from tqdm import tqdm
import copy
import re
import random

def replace_numbers_with_random_and_track(string, min_value=1, max_value=1000):
    numbers_found = False

    def replace_with_random(match):
        nonlocal numbers_found
        numbers_found = True  
        return str(random.randint(min_value, max_value)) 

    result = re.sub(r'\b\d+\b', replace_with_random, string)
    return result, numbers_found

# Load the JSON data from B.json
with open('./mydata/popQA/base/test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
def pick_random_list(lis):
    import random
    random.seed(0)
    return random.choice(lis)
# Load the BERT fill-mask pipeline
fill_mask = pipeline("fill-mask", model='roberta-large')
total = 0 
cnt = 0 
number = 0 
blank = 0
title = 0 
# Process the data
new_data = []
golden_data = []
golden_factual_error_data = []
golden_relevant_data = []
golden_irrelevant_data = []
before_factual_error_data = []
for idx, entry in enumerate(tqdm(data), start=1):

    flag = True
    samples = []
    irrelevant = 0 
    irr_samples = []
    golden = 0 
    for ctx in entry['ctxs']: 
  
        for ans in entry['answers']:
            if ans in ctx['text']:
                ctx['prediction'] = "Golden"
                samples.append(ctx)
                golden += 1
                break
   
        else:
            ctx['prediction'] = "irrelevant"
            ctx['has_answer'] = False
            irrelevant += 1
            irr_samples.append(ctx)
    if golden <2 or golden > 4: continue
    entry['ctxs'] = [e for e in entry['ctxs'] if 'prediction' in e]
    if len(entry['ctxs']) != 5: continue
    factual_error = pick_random_list(samples)
    factual_error_idx = samples.index(factual_error)
    del samples[factual_error_idx]
    evidential = pick_random_list(samples)
    irr = pick_random_list(irr_samples)
    before_factual_error = copy.deepcopy(factual_error)
    
    for answer in entry['answers']:
        if answer in evidential['text']:
            entry['answer'] = answer
            break
    else:
        continue
    for ctx in entry['ctxs']:
        if ctx['id'] == factual_error['id']:
            answers = []
            for answer in entry['answers']: 
                if answer.strip() in ctx['text'].strip():
                    answers.append(answer.strip())
            import re

            for answer in answers:
                ctx['text'] = re.sub(re.escape(answer), "<mask>", ctx['text'], flags=re.IGNORECASE)
            masked_text = ctx['text']

            question = "Question: " + entry['question']
            masked_text = question + "\n" + masked_text
            predictions = fill_mask(masked_text)
            for idx, prediction in enumerate(predictions):
                try:
                    if str(prediction['token_str']).strip() not in [str(a).strip() for a in answers]:
                        ctx['text'] = ctx['text'].replace("<mask>", str(prediction['token_str']).strip())
                        cnt += 1
                        ctx['use'] = 'roberta'
                        ctx['prediction'] = "factual_error"
                        ctx['has_answer'] = False
                        break
                except TypeError:
                    continue
            else:
                min_random = 0  
                max_random = 999  
                modified_text, modified = replace_numbers_with_random_and_track(answer.strip(), min_random, max_random)
                if modified:
                    ctx['text'] = ctx['text'].replace("<mask>", modified_text)
                    number += 1
                    ctx['use'] = 'number'
                    ctx['prediction'] = "factual_error"
                    ctx['has_answer'] = False
                    continue
                elif all(answer not in ctx['title'].strip() for answer in answers):
                    ctx['text'] = ctx['text'].replace("<mask>", ctx['title'])
                    title += 1
                    ctx['use'] = 'title'
                    ctx['prediction'] = "factual_error"
                    ctx['has_answer'] = False
                else:
                    ctx['text'] = ctx['text'].replace("<mask>", " ")
                    blank += 1
                    flag = False
    if flag:
        new_data.append(entry)
        total += 1
        g = 0
        f = 0
        ir = 0
        golden_only = [evidential]
        golden_factual = [evidential, factual_error]
        golden_irrelevant = [evidential, irr]
        double_golden = [evidential, before_factual_error]

        # Append filtered data to respective lists
        if golden_only:
            golden_data.append({**entry, "ctxs": golden_only})
        if factual_error:
            golden_factual_error_data.append({**entry, "ctxs": golden_factual})
        if golden_irrelevant:
            golden_irrelevant_data.append({**entry, "ctxs": golden_irrelevant})
        if double_golden:
            before_factual_error_data.append({**entry, 'ctxs':double_golden})


# Save each dataset to a separate file
with open('./mydata/popQA/prediction/golden_only_test.json', 'w', encoding='utf-8') as file:
    json.dump(golden_data, file, ensure_ascii=False, indent=4)

with open('./mydata/popQA/prediction/factual_error_test.json', 'w', encoding='utf-8') as file:
    json.dump(golden_factual_error_data, file, ensure_ascii=False, indent=4)

with open('./mydata/popQA/prediction/golden_irrelevant_test.json', 'w', encoding='utf-8') as file:
    json.dump(golden_irrelevant_data, file, ensure_ascii=False, indent=4)

with open('./mydata/popQA/prediction/double_golden_test.json', 'w', encoding='utf-8') as file:
    json.dump(before_factual_error_data, file, ensure_ascii=False, indent=4)
