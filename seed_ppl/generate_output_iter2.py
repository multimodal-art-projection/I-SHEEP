from utils import vllm_generator
import json
from tqdm import tqdm
import argparse

from vllm import LLM
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--seed_task_path", default='./seed_tasks.jsonl', type=str)
parser.add_argument("--meta_prompt_path", default='./prompt.txt', type=str)
parser.add_argument("--output", default='./generated.jsonl', type=str)
parser.add_argument("--gpus", default=8, type=int)
parser.add_argument("--batch_size", default=8, type=int)
parser.add_argument("--num_generation", default=10000, type=int)
args = parser.parse_args()

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }


# PROMPT_DICT = {
#     "prompt_input": (
#         "{instruction}{input}\n"
#     ),
#     "prompt_no_input": (
#         "{instruction}\n"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}{input}<|im_end|>\n<|im_start|>assistant\n"
    ),
    "prompt_no_input": (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    ),
}

# PROMPT_DICT = {
#     "prompt_input": (
#         # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#         "<|im_start|>user\nGive me three facts about the given animal. Elephant<|im_end|>\n<|im_start|>assistant\n"
#         "Sure, here are three facts about elephants:\n  1. Intelligence: Elephants are highly intelligent, capable of emotions like grief and joy, and have complex social structures led by a matriarch.\n  2. Communication: They use low-frequency sounds called infrasounds, inaudible to humans, to communicate over long distances.\n  3. Conservation Status: Elephants are endangered, with threats including habitat loss and poaching for ivory.<|im_end|>\n"
#         "<|im_start|>user\nWrite a short review of the given movie. 'The Shawshank Redemption'<|im_end|>\n<|im_start|>assistant\n"
#         "\"The Shawshank Redemption\" illustrates the power of hope and freedom through the friendship of Andy and Red in the oppressive environment of Shawshank Prison.  The film tells how Andy maintains hope and ultimately achieves freedom through resilience and intellect.  It inspires viewers that even in the darkest times, hope can be a sustaining force, and true freedom comes from an indomitable spirit.<|im_end|>\n"
#         "<|im_start|>user\nWrite an efficient, yet simple algorithm to calculate the Nth Fibonacci number. N: 12.<|im_end|>\n<|im_start|>assistant\n"
#         "An efficient and simple algorithm to compute the Nth Fibonacci number is to use an iterative approach, which significantly reduces the time complexity compared to a naive recursive approach. Here's the Python code for this algorithm:\ndef fibonacci(n):\n  a, b = 0, 1\n  if n == 0:\n    return a\n  for _ in range(2, n + 1):\n    a, b = b, a + b\n  return b\n\nfibonacci_12 = fibonacci(12)\nprint(fibonacci_12) #144<|im_end|>\n"
#         "<|im_start|>user\nTranslate the following phrase into French. I miss you.<|im_end|>\n<|im_start|>assistant\n"
#         "1. Subject: 'I' translates to 'Je'.\n2. Verb: 'miss' is 'manquer', conjugated as 'manque'.\n3. Object: 'you' is 'tu' (informal) or 'vous' (formal), but the phrase reverses to mean 'You are missed by me'.\n4. Construction: The French phrase is 'Tu me manques' (informal) or 'Vous me manquez' (formal).<|im_end|>\n"
#         "<|im_start|>user\nyou are given a list of numbers, and your task is to add them up and return the total sum. [1, 2, 3, 4, 5]<|im_end|>\n<|im_start|>assistant\n"
#         "To sum the numbers in the list [1, 2, 3, 4, 5], start at 1 and continuously add the next number: 1+2=3, 3+3=6, 6+4=10, 10+5=15. The total sum is 15.<|im_end|>\n"
#         "<|im_start|>user\n{instruction}{input}<|im_end|>\n<|im_start|>assistant\n"
#     ),
#     "prompt_no_input": (
#         # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#         "<|im_start|>user\nWhich exercises are best for reducing belly fat at home?<|im_end|>\n<|im_start|>assistant\n"
#         "  1. Lying Leg Raises\n  2. Leg In And Out\n  3. Plank\n  4. Side Plank\n  5. Sit-ups<|im_end|>\n"
#         "<|im_start|>user\nTurn down a job offer by sending an email to a recruiter explaining the reason.<|im_end|>\n<|im_start|>assistant\n"
#         "Hi  [Recruiter],\n  Thank you so much for the generous offer to join your team. As we discussed, I have admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I have decided to accept an offer at another company.\n  I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.\n  Thanks again,\n    [Your Name]<|im_end|>\n"
#         "<|im_start|>user\nWrite a program to compute the sum of integers from k to n.<|im_end|>\n<|im_start|>assistant\n"
#         "def sum(k, n):\n  sum = 0\n  for i in range(k, n+1):\n    sum += i\n  return sum<|im_end|>\n"
#         "<|im_start|>user\nRecommend a movie for me to watch during the weekend and explain the reason.<|im_end|>\n<|im_start|>assistant\n"
#         "I would recommend the movie \"The Shawshank Redemption\" because it is an excellent movie that is both moving and inspiring. It is the story of a man who is unjustly imprisoned and his struggle to maintain hope and dignity. It is a great film to watch over the weekend because it will make you think about the human capacity for resilience and hope.<|im_end|>\n"
#         "<|im_start|>user\nAdam has $100 and wants to spend it to open a rock stand. He can buy rocks for $5 each and sell them for $7 each. If he invests all his money in the rock stand but only sells 60% of his inventory, how much money does he lose?<|im_end|>\n<|im_start|>assistant\n"
#         "Adam originally has $100 and spends all of it on rocks, buying each for $5. This allows him to buy 100/5=20 rocks. He sells 60% of these rocks, which amounts to 20*0.6=12 rocks. Selling each rock at $7, he earns $7*12=$84 from these sales. Subtracting this from his initial investment of $100, Adam experiences a loss of $16.<|im_end|>\n"
#         "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
#     ),
# }

# PROMPT_DICT = {
#     "prompt_input": (
#         # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Give me three facts about the given animal. Elephant<|im_end|>\n<|im_start|>assistant's response is: "
#         "Sure, here are three facts about elephants:\n  1. Intelligence: Elephants are highly intelligent, capable of emotions like grief and joy, and have complex social structures led by a matriarch.\n  2. Communication: They use low-frequency sounds called infrasounds, inaudible to humans, to communicate over long distances.\n  3. Conservation Status: Elephants are endangered, with threats including habitat loss and poaching for ivory.<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Write a short review of the given movie. 'The Shawshank Redemption'<|im_end|>\n<|im_start|>assistant's response is: "
#         "\"The Shawshank Redemption\" illustrates the power of hope and freedom through the friendship of Andy and Red in the oppressive environment of Shawshank Prison.  The film tells how Andy maintains hope and ultimately achieves freedom through resilience and intellect.  It inspires viewers that even in the darkest times, hope can be a sustaining force, and true freedom comes from an indomitable spirit.<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Write an efficient, yet simple algorithm to calculate the Nth Fibonacci number. N: 12.<|im_end|>\n<|im_start|>assistant's response is: "
#         "An efficient and simple algorithm to compute the Nth Fibonacci number is to use an iterative approach, which significantly reduces the time complexity compared to a naive recursive approach. Here's the Python code for this algorithm:\ndef fibonacci(n):\n  a, b = 0, 1\n  if n == 0:\n    return a\n  for _ in range(2, n + 1):\n    a, b = b, a + b\n  return b\n\nfibonacci_12 = fibonacci(12)\nprint(fibonacci_12) #144<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Translate the following phrase into French. I miss you.<|im_end|>\n<|im_start|>assistant's response is: "
#         "1. Subject: 'I' translates to 'Je'.\n2. Verb: 'miss' is 'manquer', conjugated as 'manque'.\n3. Object: 'you' is 'tu' (informal) or 'vous' (formal), but the phrase reverses to mean 'You are missed by me'.\n4. Construction: The French phrase is 'Tu me manques' (informal) or 'Vous me manquez' (formal).<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: You are given a list of numbers, and your task is to add them up and return the total sum. [1, 2, 3, 4, 5]<|im_end|>\n<|im_start|>assistant's response is: "
#         "To sum the numbers in the list [1, 2, 3, 4, 5], start at 1 and continuously add the next number: 1+2=3, 3+3=6, 6+4=10, 10+5=15. The total sum is 15.<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: {instruction}{input}<|im_end|>\n<|im_start|>assistant's response is: "
#     ),
#     "prompt_no_input": (
#         # "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Which exercises are best for reducing belly fat at home?<|im_end|>\n<|im_start|>assistant's response is: "
#         "  1. Lying Leg Raises\n  2. Leg In And Out\n  3. Plank\n  4. Side Plank\n  5. Sit-ups<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Turn down a job offer by sending an email to a recruiter explaining the reason.<|im_end|>\n<|im_start|>assistant's response is: "
#         "Hi  [Recruiter],\n  Thank you so much for the generous offer to join your team. As we discussed, I have admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I have decided to accept an offer at another company.\n  I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.\n  Thanks again,\n    [Your Name]<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Write a program to compute the sum of integers from k to n.<|im_end|>\n<|im_start|>assistant's response is: "
#         "def sum(k, n):\n  sum = 0\n  for i in range(k, n+1):\n    sum += i\n  return sum<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Recommend a movie for me to watch during the weekend and explain the reason.<|im_end|>\n<|im_start|>assistant's response is: "
#         "I would recommend the movie \"The Shawshank Redemption\" because it is an excellent movie that is both moving and inspiring. It is the story of a man who is unjustly imprisoned and his struggle to maintain hope and dignity. It is a great film to watch over the weekend because it will make you think about the human capacity for resilience and hope.<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: Adam has $100 and wants to spend it to open a rock stand. He can buy rocks for $5 each and sell them for $7 each. If he invests all his money in the rock stand but only sells 60% of his inventory, how much money does he lose?<|im_end|>\n<|im_start|>assistant's response is: "
#         "Adam originally has $100 and spends all of it on rocks, buying each for $5. This allows him to buy 100/5=20 rocks. He sells 60% of these rocks, which amounts to 20*0.6=12 rocks. Selling each rock at $7, he earns $7*12=$84 from these sales. Subtracting this from his initial investment of $100, Adam experiences a loss of $16.<|im_end|><|endoftext|>\n"
#         "<|im_start|>user\nYou are a helpful assistant. please complete the following task directly. Task: {instruction}<|im_end|>\n<|im_start|>assistant's response is: "
#     ),
# }

# def encode_rawoutput(prompt):
#     raw_output = '''For the given new task, please generate the output directly.
    
#     Task: Which exercises are best for reducing belly fat at home?
#     Output:
#     - Lying Leg Raises
#     - Leg In And Out
#     - Plank
#     - Side Plank
#     - Sit-ups

#     Task: Converting 85 F to Celsius.
#     Output: 85°F = 29.44°C

#     Task: Write a program to compute the sum of integers from k to n.
#     Output:
#     def sum(k, n):
#         sum = 0
#         for i in range(k, n+1):
#             sum += i
#         return sum

#     Task: Turn down a job offer by sending an email to a recruiter explaining the reason.
#     Output: Hi  [Recruiter],
#     Thank you so much for the generous offer to join your team. As we discussed, I’ve admired the company for a number of years, and am a proud endorser of its products. However, after further consideration of where I currently am in my career, I’ve decided to accept an offer at another company.
#     I would love to stay in touch with you and have already started following you on [Social Media Platform]. Again, thank you so much for your time and consideration.
#     Thanks again,
#     [Your Name]

#     Task: Recommend a movie for me to watch during the weekend and explain the reason.
#     Output: I would recommend the movie \"The Shawshank Redemption\" because it is an excellent movie that is both moving and inspiring. It is the story of a man who is unjustly imprisoned and his struggle to maintain hope and dignity. It is a great film to watch over the weekend because it will make you think about the human capacity for resilience and hope.

#     Task: Find the four smallest perfect numbers.
#     Output: 6, 28, 496, 8128

#     Task:'''
#     return f"{raw_output}+{prompt}\nOutput:"

# def encode_input(prompt):
#     ### sample from alpaca dataset
#     with_instruction = """For the given new instruction and input pair, please generate the output directly.
    
#     instruction: Correct this sentence for spelling and grammar mistakes.
#     input: He finnished his meal and left resturant.
#     output: He finished his meal and left the restaurant.
    
#     instruction: Summarize the given passage.
#     input: A recent study showed that global climate change is one of the most important challenges facing the world today. The consequences of global warming include rising sea levels, extreme weather events and biodiversity loss. Climate change is caused by human activities such as burning fossil fuels, and if unchecked will have long-term and devastating impacts on the planet.
#     output: A recent study revealed that global climate change is one of the world\u2019s most pressing issues. In addition to rising sea levels, extreme weather events and biodiversity loss, it is caused largely by human activities like burning fossil fuels. If left unchecked, climate change will have far-reaching and devastating consequences.
    
#     instruction: Arrange the given numbers in ascending order.
#     input: 2, 4, 0, 8, 3.
#     output: 0, 2, 3, 4, 8.

#     instruction: Translate the following phrase into French.
#     input: I miss you
#     output: Je te manque.

#     instruction: Classify the following statement as true or false.
#     input: The Supreme Court is the highest court in the US.
#     output: True.

#     instruction:
#     """
#     ins = prompt["instruction"]
#     inp = prompt["input"]
#     return f"{with_instruction}{ins}\ninput: {inp}\noutput: "

prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]

# generation_config = GenerationConfig.from_pretrained(args.model,trust_remote_code=True)
# # 加载分词器
# tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True)
# tokenizer.eos_token_id=generation_config.eos_token_id
# # 推理终止词，遇到这些词停止继续推理
# stop_words_ids=[tokenizer.im_start_id,tokenizer.im_end_id,tokenizer.eos_token_id]
# "<|endoftext|>"

def encode_rawoutput(prompt):
    raw_output = prompt_no_input
    # print("Prompt:", raw_output.format_map({'instruction':prompt}), "\n\n")
    
    return raw_output.format_map({'instruction':prompt})

def encode_input(prompt):
    ### sample from alpaca dataset
    with_instruction = prompt_input
    ins = prompt["instruction"]
    inp = prompt["input"]
    
    # print("Prompt:", with_instruction.format_map({'instruction':ins, 'input':inp}), "\n\n")
    return with_instruction.format_map({'instruction':ins, 'input':inp})

def main(model, size, seed_tasks_path, request_batch_size, targetfp):
    with open(seed_tasks_path, "r", encoding='utf-8') as f:
        seed_tasks = [json.loads(l) for l in f]
    with open(targetfp, "r", encoding='utf-8') as f:
        target_tasks = [json.loads(l) for l in f]
    seed_instruction_data = [
        {
            "instruction": t["instruction"],
            "input": t["input"],
            "output": t["output"],
            "ppl":t['ppl'] if 'ppl' in t.keys() else -100
        } for t in seed_tasks
    ]
    print('all len', len(seed_instruction_data))
    seed_instruction_data = seed_instruction_data[len(target_tasks):] 
    print('generated len', len(target_tasks))
    print('not generated len', len(seed_instruction_data))
    llm = vllm_generator(model, size, ["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    raw_instruction_data = []
    with_input = []

    for tasks in seed_instruction_data:
        if  tasks["input"]=="":
            raw_instruction_data.append(tasks)
        else:
            with_input.append(tasks)
    f_results = []

    
    # for i in range(0, len(raw_instruction_data), request_batch_size):
    for i in tqdm(range(0, len(raw_instruction_data), request_batch_size), desc="Processing raw_instruction_data"):
            if i+request_batch_size>=len(raw_instruction_data):
                batch = raw_instruction_data[i:]
            else:
                batch = raw_instruction_data[i:i + request_batch_size]
            batch_inputs = []
            for l in batch:
                tmp_prompt=encode_rawoutput(l["instruction"])
                print(tmp_prompt)
                batch_inputs.append(tmp_prompt)
            # print(batch_inputs)
            results = llm.generate(batch_inputs)
            # outputs = llm.generate(batch_inputs, sampling_params)
            # results = [output.outputs[0].text for output in outputs]
            for i in range(len(batch)):
                batch[i]["output"] = results[i]
            f_results += batch
    

    for i in tqdm(range(0, len(with_input), request_batch_size), desc="Processing with_input"):
            if i+request_batch_size>=len(with_input):
                batch = with_input[i:]
            else:
                batch = with_input[i:i + request_batch_size]
            batch_inputs = []
            for l in batch:
                tmp_prompt = encode_input(l)
                print(tmp_prompt)
                batch_inputs.append(tmp_prompt)
            results = llm.generate(batch_inputs)
            for i in range(len(batch)):
                batch[i]["output"] = results[i]
            f_results += batch
    output_handler = open(targetfp, "a", encoding='utf-8')
    for result in seed_instruction_data:
            output_handler.write(json.dumps(result, ensure_ascii=False))
            output_handler.write("\n")           
    output_handler.flush()


    
    
# prompt = {'instruction':'123', 'input':'234'}
# print(encode_rawoutput(prompt['instruction']))
main(args.model , args.gpus, args.seed_task_path, args.batch_size, args.output)

# main("/ML-A100/home/gezhang/models/llama1-65B-hf", 8, "/ML-A100/home/gezhang/ppl_test/ppl_10k.jsonl", 100, "output_10k.jsonl")
# main("/ML-A100/home/gezhang/models/llama1-65B-hf", 8, "/ML-A100/home/gezhang/ppl_test/noppl_10k.jsonl", 100, "output_10k_noppl.jsonl")