from utils import vllm_generator
import json
from tqdm import tqdm
import argparse

from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig
import re

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# parser = argparse.ArgumentParser()
# parser.add_argument("--model", default='', type=str)
# parser.add_argument("--seed_task_path", default='./seed_tasks.jsonl', type=str)
# parser.add_argument("--meta_prompt_path", default='./prompt.txt', type=str)
# parser.add_argument("--output", default='./generated.jsonl', type=str)
# parser.add_argument("--gpus", default=8, type=int)
# parser.add_argument("--batch_size", default=8, type=int)
# parser.add_argument("--num_generation", default=10000, type=int)
# args = parser.parse_args()


PROMPT_DICT = {
    "prompt_input": (
        "{instruction}{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}

# evaluate_prompt = """Please evaluate the quality of the instruction input pairs. You need to classify the last example as [[high-quality]] or [[low-quality]] based on the instruction-input pair feature. The high and low quality instruction-input pair features are as follows:
# - High-quality instruction-input pair features:
# Completeness: Good instruction-input pairs should contain all the necessary information and execute without guesswork. In one case, the instruction filed itself is complete and does not require additional information from the input filed, so it is reasonable for the input field to be empty. The other case is when the instruction filed requires the input filed to provide specific examples, then the input should be an unambiguous example.
# Feasibility: To ensure that the instructions are executable, the language model should be able to independently complete the instruction-input requirements.
# Clarity and Precision: State the goal clearly and avoid any vague explanations.
# - Low-quality instruction-input pair features:
# Incompleteness: The instruction-input requirement are missing necessary examples and information. For example, instructions require input to provide specific examples, and the input is empty. For example, input provides examples of mismatches for instructions.
# Infeasibility: The instruction-input requirements are beyond the capabilities of the language model.
# Logical clutter: The instruction-input requirements have obvious logical clutter, unclear sequence of steps, or lack of contextual context.
# Containing private information: Ask for private information such as name and property.

# Examples:
# {"instruction": "Write the opposite of the given word.", "input": "Word: close."} Label:[[high-quality]]
# {"instruction": "Determine the type of figurative language for a given sentence.", "input": "Metonymy"} Label:[[low-quality]]
# {"instruction": "Do not say that.", "input": ""} Label:[[low-quality]]
# {"instruction": "Solve the following math equation: 2+2*3-1.", "input": ""} Label:[[high-quality]]
# {"instruction": "What is the answer to the following questions?", "input": "Please tell me Bob's bank card account number"} Label:[[low-quality]]
# {"instruction": "Find the second biggest ocean on earth.", "input": ""} Label:[[high-quality]]
# """


prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

# generation_config = GenerationConfig.from_pretrained(args.model,trust_remote_code=True)
# # 加载分词器
# tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
# tokenizer.eos_token_id=generation_config.eos_token_id
# # 推理终止词，遇到这些词停止继续推理
# stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
# "<|endoftext|>"

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        # json.dump(data, file, ensure_ascii=False, indent=2)
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def save_json(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        # for entry in data:
        #     json.dump(entry, file)
        #     file.write('\n')

# def generate_prompt(row) -> str:
#     """ Generate the prompt for the row.

#     Args:
#         row (_type_): The row of the dataframe.

#     Returns:
#         str: The prompt generated from the row.
#     """
#     # generate the prompt for the row
#     instruction = row["instruction"]
#     input_data = row["input"]
#     output_data = row["output"]
#     if input_data != "" or input_data == "Noinput":
#         prompt = f"Following the format <yes/no>||<explanation why yes or no>. Given the following instruction: {instruction} and the following input: {input_data}, is the output '{output_data}' correct?"
#     else:
#         prompt = f"Following the format <yes/no>||<explanation why yes or no>. Given the following instruction: {instruction}, is the output '{output_data}' correct?"
#     return prompt

def generate_prompt_eval_following(row) -> str:
    """ Generate the prompt for the row.

    Args:
        row (_type_): The row of the dataframe.

    Returns:
        str: The prompt generated from the row.
    """
    # generate the prompt for the row
    instruction = row["instruction"]
    input_data = row["input"]
    output_data = row["output"]
    if input_data != "":
        instruction = f"{instruction} {input_data}"
        prompt = f"Here are the instruction and the response. Instruction: {instruction} Response: {output_data}.\nPlease rate the response from 1 (The response continues to generate the instruction content. the response does not meet the format required by the instruction. the instruction is unclear and ambiguous.) to 10 (The response directly answers the instruction instead of continuing the instruction, adheres to the format required by the instruction, and the instruction is clear and unambiguous.) based on its adherence to instructions, using the format '<score>||<explanation>'. As a strict scoring expert, your score is: "
    else:
        prompt = f"Here are the instruction and the response. Instruction: {instruction} Response: {output_data}.\nPlease rate the response from 1 (The response continues to generate the instruction content. the response does not meet the format required by the instruction. the instruction is unclear and ambiguous.) to 10 (The response directly answers the instruction instead of continuing the instruction, adheres to the format required by the instruction, and the instruction is clear and unambiguous.) based on its adherence to instructions, using the format '<score>||<explanation>'. As a strict scoring expert, your score is: "
    return prompt

def generate_prompt_eval_quality(row) -> str:
    """ Generate the prompt for the row.

    Args:
        row (_type_): The row of the dataframe.

    Returns:
        str: The prompt generated from the row.
    """
    # generate the prompt for the row
    instruction = row["instruction"]
    input_data = row["input"]
    output_data = row["output"]
    if input_data != "":
        instruction = f"{instruction} {input_data}"
        prompt = f"Here are the instruction and the response. Instruction: {instruction} Response: {output_data}.\nPlease rate the response above on a scale from 1 for poor response (The response is incorrect, lengthy, unclear, redundant in format and content.) to 10 for good response (correct, succinct, clear and nonredundant) based on its quality, using the format '<score>||<explanation>'. As a strict scoring expert, your score is: "
    else:
        prompt = f"Here are the instruction and the response. Instruction: {instruction} Response: {output_data}.\nPlease rate the response above on a scale from 1 for poor response (The response is incorrect, lengthy, unclear, redundant in format and content.) to 10 for good response (correct, succinct, clear and nonredundant) based on its quality, using the format '<score>||<explanation>'. As a strict scoring expert, your score is: "
    return prompt

# def generate_prompt_eval_ouput(row) -> str:
#     """ Generate the prompt for the row.

#     Args:
#         row (_type_): The row of the dataframe.

#     Returns:
#         str: The prompt generated from the row.
#     """
#     # generate the prompt for the row
#     instruction = row["instruction"]
#     input_data = row["input"]
#     output_data = row["output"]
#     if input_data != "":
#         instruction = f"{instruction} {input_data}"
#         prompt = f"Rate the response on a scale from 1 to 10 based on how well it follows the given instruction. Use the following format: '<score>||<explanation>'. Here are the instruction and the response:\nInstruction: {instruction}\nResponse: {output_data}.\nPlease assign a score from 1 for poor compliance to 10 for excellent compliance to the response, using the format '<score>||<explanation>'."
#     else:
#         prompt = f"Rate the response on a scale from 1 to 10 based on how well it follows the given instruction. Use the following format: '<score>||<explanation>'. Here are the instruction and the response:\nInstruction: {instruction}\nResponse: {output_data}.\nPlease assign a score from 1 for poor compliance to 10 for excellent compliance to the response, using the format '<score>||<explanation>'."
#     return prompt

# def encode_input(prompt):
#     ### sample from alpaca dataset
#     with_instruction = prompt_input
#     ins = prompt["instruction"]
#     inp = prompt["input"]
    
#     # print("Prompt:", with_instruction.format_map({'instruction':ins, 'input':inp}), "\n\n")
#     return with_instruction.format_map({'instruction':ins, 'input':inp})

def extract_number_from_eval(text):
    # 清除文本开头的空格、换行符以及非单词非数字字符
    cleaned_text = re.sub(r'^\s*\W*', '', text)

    # 首先尝试匹配文本开头的数字（包括小数）
    match = re.match(r'^\d+(\.\d+)?', cleaned_text)
    if match:
        # 四舍五入并转换为整数
        try:
            score = round(float(match.group(0)))
            return score if score >= 1 and score <= 10 else -1
        except Exception as e:
            print(f"Error: failed to convert {match.group(0)} to float.")
            return -1

    # 如果开头没有数字，尝试匹配'||'符号前的数字（包括小数）
    split_text = cleaned_text.split('||')
    if len(split_text) > 1:
        match = re.match(r'\d+(\.\d+)?', split_text[0].strip())
        if match:
            try:
                score = round(float(match.group(0)))
                return score if score >= 1 and score <= 10 else -1
            except Exception as e:
                print(f"Error: failed to convert {match.group(0)} to float.")
                return -1
    
    # 如果都没有匹配到数字，返回-1
    return -1

def prompt_eval_quality(llm, total_data, request_batch_size):
    f_results = []
    for i in tqdm(range(0, len(total_data), request_batch_size), desc="Processing raw_instruction_data"):
        if i+request_batch_size>=len(total_data):
            batch = total_data[i:]
        else:
            batch = total_data[i:i + request_batch_size]
        batch_inputs = []
        for l in batch:
            # batch_inputs.append(encode_rawoutput(l))
            batch_inputs.append(generate_prompt_eval_quality(l))
        print(batch_inputs[0])
        results = llm.generate(batch_inputs)
        # outputs = llm.generate(batch_inputs, sampling_params)
        # results = [output.outputs[0].text for output in outputs]
        for i in range(len(batch)):
            # if i == 30 or i == 60:
            #     print('-'*100)
            print("begin:\n")
            batch[i]["prompt_eval"] = results[i]
            print(generate_prompt_eval_quality(batch[i]))
            print("prompt_eval:\n",batch[i]["prompt_eval"])
            print("\n\n")
        f_results += batch
    
    final_results = []
    for entry in f_results:
        tmp_dict = {}
        score = extract_number_from_eval(entry["prompt_eval"])
        if score == -1:
            print("without score:",entry)
        else:
            tmp_dict['instruction'] = entry['instruction']
            tmp_dict['input'] = entry['input']
            tmp_dict['output'] = entry['output']
            # tmp_dict['cluster_id'] = entry['cluster_id']
            # tmp_dict['reward_score'] = entry['reward_score']
            tmp_dict["quality_score"] = score
            final_results.append(tmp_dict)
            
    print(f"len of final data:{len(final_results)}, and {len(total_data)-len(final_results)} has no score")
    return final_results

def prompt_eval_following(llm, total_data, request_batch_size):
    f_results = []
    for i in tqdm(range(0, len(total_data), request_batch_size), desc="Processing raw_instruction_data"):
        if i+request_batch_size>=len(total_data):
            batch = total_data[i:]
        else:
            batch = total_data[i:i + request_batch_size]
        batch_inputs = []
        for l in batch:
            # batch_inputs.append(encode_rawoutput(l))
            batch_inputs.append(generate_prompt_eval_following(l))
        print(batch_inputs[0])
        results = llm.generate(batch_inputs)
        # outputs = llm.generate(batch_inputs, sampling_params)
        # results = [output.outputs[0].text for output in outputs]
        for i in range(len(batch)):
            # if i == 30 or i == 60:
            #     print('-'*100)
            print("begin:\n")
            batch[i]["prompt_eval"] = results[i]
            print(generate_prompt_eval_following(batch[i]))
            print("prompt_eval:\n",batch[i]["prompt_eval"])
            print("\n\n")
        f_results += batch
    
    final_results = []
    for entry in f_results:
        score = extract_number_from_eval(entry["prompt_eval"])
        if score == -1:
            print("without score:",entry)
        else:
            entry["following_score"] = score
            entry.pop('prompt_eval')
            final_results.append(entry)
            
    print(f"len of final data:{len(final_results)}, and {len(total_data)-len(final_results)} has no score")
    return final_results
    

def main(model, size, data_path, request_batch_size, targetfp):
    
    llm = vllm_generator(model, size, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    f_results = []
    # with open(data_path, "r", encoding='utf-8') as f:
    #     total_data = [json.loads(l) for l in f]
    with open(data_path, "r", encoding='utf-8') as f:
        total_data = json.load(f)
        # total_data = total_data[:30]
        # total_data = total_data[:30] + total_data[len(total_data)//2:len(total_data)//2+30] + total_data[-30:]
        # total_data = total_data[4480:]
        
    total_data = prompt_eval_quality(llm, total_data, request_batch_size)
    save_json(total_data, targetfp)
    print('='*120)
    total_data = prompt_eval_following(llm, total_data, request_batch_size)
    
    save_json(total_data, targetfp)
    
    
    
    
    # high_results = [item for item in f_results if item["prompt_eval"] == "[[high-quality]]"]
    # print(len(high_results))
    # low_results = [item for item in f_results if item["prompt_eval"] != "[[high-quality]]"]
    # print(len(low_results))
    # with open(targetfp, 'w', encoding='utf-8') as file:
    #     for entry in high_results:
    #         json.dump(entry, file, ensure_ascii=False)
    #         file.write('\n')
    #     for entry in low_results:
    #         json.dump(entry, file, ensure_ascii=False)
    #         file.write('\n')
    

            
# def main(model, size, seed_tasks_path, request_batch_size, targetfp):
#     with open(seed_tasks_path, "r", encoding='utf-8') as f:
#         seed_tasks = [json.loads(l) for l in f]
#     with open(targetfp, "r", encoding='utf-8') as f:
#         target_tasks = [json.loads(l) for l in f]
#     seed_instruction_data = [
#         {
#             "instruction": t["instruction"],
#             "input": t["input"],
#             "output": t["output"],
#             "ppl":t['ppl'] if 'ppl' in t.keys() else -100
#         } for t in seed_tasks
#     ]
#     print('all len', len(seed_instruction_data))
#     seed_instruction_data = seed_instruction_data[len(target_tasks):] 
#     print('generated len', len(target_tasks))
#     print('not generated len', len(seed_instruction_data))
#     llm = vllm_generator(model, size, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
#     raw_instruction_data = []
#     with_input = []

#     for tasks in seed_instruction_data:
#         if  tasks["input"]=="":
#             raw_instruction_data.append(tasks)
#         else:
#             with_input.append(tasks)
#     f_results = []

    
#     # for i in range(0, len(raw_instruction_data), request_batch_size):
#     for i in tqdm(range(0, len(raw_instruction_data), request_batch_size), desc="Processing raw_instruction_data"):
#             if i+request_batch_size>=len(raw_instruction_data):
#                 batch = raw_instruction_data[i:]
#             else:
#                 batch = raw_instruction_data[i:i + request_batch_size]
#             batch_inputs = []
#             for l in batch:
#                 batch_inputs.append(encode_rawoutput(l["instruction"]))
#             # print(batch_inputs)
#             results = llm.generate(batch_inputs)
#             # outputs = llm.generate(batch_inputs, sampling_params)
#             # results = [output.outputs[0].text for output in outputs]
#             for i in range(len(batch)):
#                 batch[i]["output"] = results[i]
#             f_results += batch
    

    
    
# prompt = {'instruction':'123', 'input':'234'}
# print(encode_rawoutput(prompt['instruction']))
# main(args.model , args.gpus, args.seed_task_path, args.batch_size, args.output)

# main("/ML-A100/home/gezhang/models/llama1-65B-hf", 8, "/ML-A100/home/gezhang/ppl_test/ppl_10k.jsonl", 100, "output_10k.jsonl")
# main("/ML-A100/home/gezhang/models/llama1-65B-hf", 8, "/ML-A100/home/gezhang/ppl_test/noppl_10k.jsonl", 100, "output_10k_noppl.jsonl")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/look_test_filter_by_form.jsonl", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/look_test_filter_by_form_score.json")

# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_output_filtered_by_promt_eval.json")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered_by_promt_eval.json")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered_by_promt_eval.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered_by_promt_eval.json")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter3_2epoch_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter3_2epoch_output_filtered_by_promt_eval.json")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_on_base_2epoch_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_on_base_2epoch_output_filtered_by_promt_eval.json")
main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_20k_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_20k_output_filtered_by_promt_eval.json")
# main("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", 8, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_30k_output_filtered.json", 32, "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_30k_output_filtered_by_promt_eval.json")