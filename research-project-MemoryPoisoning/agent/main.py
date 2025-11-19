import json
import random
import argparse
import time
import asyncio
import traceback

import numpy as np
from autogen_ext.models.openai import OpenAIChatCompletionClient

from toolset_high import *
from medagent import MedAgent


def judge(pred, ans):
    old_flag = True
    if not ans in pred:
        old_flag = False
    if "True" in pred:
        pred = pred.replace("True", "1")
    else:
        pred = pred.replace("False", "0")
    if ans == "False" or ans == "false":
        ans = "0"
    if ans == "True" or ans == "true":
        ans = "1"
    if ans == "No" or ans == "no":
        ans = "0"
    if ans == "Yes" or ans == "yes":
        ans = "1"
    if ans == "None" or ans == "none":
        ans = "0"
    if ", " in ans:
        ans = ans.split(', ')
    if ans[-2:] == ".0":
        ans = ans[:-2]
    if not type(ans) == list:
        ans = [ans]
    new_flag = True
    for i in range(len(ans)):
        if not ans[i] in pred:
            new_flag = False
            break
    return (old_flag or new_flag)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mimic_iii")
    parser.add_argument("--data_path", type=str, default="<YOUR_DATASET_PATH>")
    parser.add_argument("--logs_path", type=str, default="<YOUR_LOGS_PATH>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_id", type=str, default="521fd2885f51641a963f8d3e")
    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--num_shots", type=int, default=4)
    args = parser.parse_args()
    set_seed(args.seed)
    if args.dataset == 'mimic_iii':
        from prompts_mimic import EHRAgent_4Shots_Knowledge
    else:
        from prompts_eicu import EHRAgent_4Shots_Knowledge

    # config_list = [openai_config(args.llm)]
    # llm_config = llm_config_list(args.seed, config_list)

    model_client = OpenAIChatCompletionClient(
        model="gemini-2.0-flash-exp",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="AIzaSyD5IDnspZ_2ha0PPLGzaTHAcfoR2ixzTis",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
        },
    )

    user_proxy = MedAgent(
        name="user_proxy",
        model_client=model_client,
        dataset=args.dataset,
        system_message="You are a helpful EHR coding agent.",
    )


    file_path = args.data_path
    # read from json file
    with open(file_path, 'r') as f:
        contents = json.load(f)

    # random shuffle
    import random
    random.shuffle(contents)
    file_path = "{}/{}/".format(args.logs_path, args.num_shots) + "{id}.txt"

    start_time = time.time()
    if args.num_questions == -1:
        args.num_questions = len(contents)
    long_term_memory = []
    init_memory = EHRAgent_4Shots_Knowledge
    init_memory = init_memory.split('\n\n')
    print("-----------Initial Long-Term Memory-----------")
    for i in range(len(init_memory)):
        item = init_memory[i]
        print(item)
        item = item.split('Question:')[-1]
        question = item.split('\nKnowledge:\n')[0]
        item = item.split('\nKnowledge:\n')[-1]
        knowledge = item.split('\nSolution:')[0]
        code = item.split('\nSolution:')[-1]
        new_item = {"question": question, "knowledge": knowledge, "code": code}
        long_term_memory.append(new_item)

    for i in range(args.start_id, args.num_questions):
        print(contents[i])
        if args.debug and contents[i]['id'] != args.debug_id:
            continue
        question = contents[i]['template']
        answer = contents[i]['answer']
        try:
            user_proxy.update_memory(args.num_shots, long_term_memory)
            response, logs_string = await user_proxy.answer(question)
            print("-----------Question-----------")
            print(question)
            print("-----------Response-----------")
            print(response)
            print("-----------End of Response-----------")
        except Exception as e:
            traceback.print_exc()
            logs_string = [traceback.format_exc()]
        print(logs_string)
        file_directory = file_path.format(id=contents[i]['id'])
        if type(answer) == list:
            answer = ', '.join(answer)
        logs_string.append("Ground-Truth Answer ---> "+answer)
        with open(file_directory, 'w') as f:
            f.write('\n----------------------------------------------------------\n'.join(logs_string))
        logs_string = '\n----------------------------------------------------------\n'.join(logs_string)
        if '"cell": "' in logs_string:
            last_code_start = logs_string.rfind('"cell": "')
            last_code_end = logs_string.rfind('"\n}')
            last_code = logs_string[last_code_start+9:last_code_end]
        else:
            last_code_end = logs_string.rfind('Solution:')
        prediction_end = logs_string.rfind('TERMINATE')
        prediction = logs_string[last_code_end:prediction_end]
        result = judge(prediction, answer)
        if result:
            new_item = {"question": question, "knowledge": user_proxy.knowledge, "code": user_proxy.code}
            long_term_memory.append(new_item)
    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)

if __name__ == "__main__":
    asyncio.run(main())