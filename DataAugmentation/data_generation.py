
from openai import OpenAI
from rich.console import Console
from typing import List, Optional
import json
from tqdm import tqdm
import argparse
import os
from collections import defaultdict
class GPTchatClass:
    def __init__(
            self,
            gpt_model: str = "gpt-3.5-turbo",
            role_msg: str = "Your are a helpful assistant.",
            key_path: str = '<Your API Key>',
    ):
        self.gpt_model = gpt_model
        self.role_msg = role_msg
        self.key_path = key_path

        self.messages = [{"role": "system", "content": f"{self.role_msg}"}]
        self.init_messages = [{"role": "system", "content": f"{self.role_msg}"}]
        self.response = None
        self.console = Console()

        self._setup_client()

    def _setup_client(self):
        with open(self.key_path, "r") as f:
            OPENAI_API_KEY = f.read()
        self.client = OpenAI(api_key=OPENAI_API_KEY)

    def _add_message(
            self,
            role="assistant",
            content="",
    ):
        """
        role: 'assistant' / 'user'
        """
        self.messages.append({"role": role, "content": content})


    def _get_response_content(self):
        if self.response:
            return self.response.choices[0].message.content
        else:
            return None

    def _get_response_status(self):
        if self.response:
            return self.response.choices[0].message.finish_reason
        else:
            return None

    def reset(
            self
    ):
        self.init_messages = [{"role": "system", "content": f"{self.role_msg}"}]
        self.messages = self.init_messages

    def chat(
            self,
            user_msg="hi",
            PRINT_USER_MSG=True,
            PRINT_GPT_OUTPUT=True,
            RESET_CHAT=False,
            RETURN_RESPONSE=False,
    ):
        self._add_message(role="user", content=user_msg)
        self.response = self.client.chat.completions.create(
            messages=self.messages,
            model=self.gpt_model,
        )
        # Backup response for continous chatting
        self._add_message(role="assistant", content=self._get_response_content())
        if PRINT_USER_MSG:
            self.console.print("[deep_sky_blue3][USER_MSG][/deep_sky_blue3]")

        if PRINT_GPT_OUTPUT:
            self.console.print("[spring_green4][GPT_OUTPUT][/spring_green4]")
        # Reset
        if RESET_CHAT:
            self.reset()
        # Return
        if RETURN_RESPONSE:
            return self._get_response_content()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default="./mydata/tqa/temp/dev.json")
    parser.add_argument('--output_dir', type=str, default="./mydata/tqa/summarization/")
    parser.add_argument('--out_file_name', type=str, default="dev")
    args = parser.parse_args()


    with open(args.input_dir, "r", encoding='utf-8') as fin:
        datasets = []
    
        for line in fin:
            if line.strip():
                try:
                    data = json.loads(line)
                    datasets.append(data)
                except json.JSONDecodeError as e:
                    print(f"JSONDecodeError: {e} for line: {line}")
    var_info = defaultdict(int)
    ANS = GPTchatClass(
            gpt_model="gpt-3.5-turbo",  # 'gpt-3.5-turbo' / 'gpt-4'
            role_msg="Your are a helpful assistant. Compress the information in the retrieved documents into a 2-sentence summary that could be used to answer the question:\n",
            key_path="./key/temp_key.txt",
            )
    eval_out = open(os.path.join(args.output_dir, args.out_file_name+".json"), "w", encoding="utf-8")
    for idx, data in enumerate(tqdm(datasets[2968:40000])):
        question = data['question']
        answers = data['answers']

        if data['evidentiality_cnt'] == 0:
            data['summary'] = ""
            var_info[data['evidentiality_cnt']] += 1
        else:
            docs = [data['ctxs'][idx]['text'] for idx in data['evidentiality_idx']]
            user_msg = "\n".join(docs)
            user_msg += "\n\n" + "Question: " + question + "\nSummary: "
            PRINT_USER_MSG = False
            PRINT_GPT_OUTPUT = False
            RESET_CHAT = True
            RETURN_RESPONSE = True
            out = ANS.chat(
                user_msg=user_msg, PRINT_USER_MSG=PRINT_USER_MSG, PRINT_GPT_OUTPUT=PRINT_GPT_OUTPUT,
                RESET_CHAT=RESET_CHAT, RETURN_RESPONSE=RETURN_RESPONSE)
            data['summary'] = out
            var_info[data['evidentiality_cnt']] += 1
        eval_out.write(json.dumps(data, ensure_ascii=False)+"\n")
    eval_out.close()
    # Save log information
    log_file = open(os.path.join(args.output_dir, args.out_file_name)+".txt", "w", encoding='utf-8')
    log_file.write(f"the total number of data {idx+1}, zero: {var_info[0]}, one: {var_info[1]}, two: {var_info[2]}, three: {var_info[3]}, four: {var_info[4]}, five: {var_info[5]}")
    log_file.close()