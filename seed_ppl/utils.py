import torch
import numpy as np
from vllm import LLM, SamplingParams

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from transformers import GenerationConfig
from transformers.generation import GenerationConfig

import time

class hf_generator():
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            model_path, 
            padding_side="left", 
            trust_remote_code=True)
        self.tokenizer.pad_token_id = 0 if self.tokenizer.pad_token_id is None else self.tokenizer.pad_token_id
        self.model = LlamaForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        self.model.to_bettertransformer()

    def generate(self, batch_inputs, max_length):
        batch = self.tokenizer(
            batch_inputs,
            padding="longest",
            return_tensors="pt",
        )
        output_ids = self.model.generate(
            batch.input_ids.to(self.model.device),
            attention_mask=batch.input_ids.ne(self.tokenizer.pad_token_id).to(self.model.device),
            pad_token_id = self.tokenizer.pad_token_id,
            labels = batch.input_ids.clone().to(self.model.device) ,
            generation_config=GenerationConfig(do_sample=False, max_new_tokens=max_length, trust_remote_code=True),
            no_repeat_ngram_size=5
        )
        tokens = []
        for output_id in output_ids.tolist():
            #print("==" * 10) 
            tmp = self.tokenizer.decode(output_id[batch.input_ids.shape[-1]:], skip_special_tokens=True)
            tokens.append(tmp)
            #print(tmp)
        return tokens


class vllm_generator():
    def __init__(self, model,size, stops):
        self.generator = LLM(model, tensor_parallel_size=size,trust_remote_code=True)
        self.sampling_params = SamplingParams(temperature=1, top_p=0.95, stop=stops, max_tokens = 1000)
        
    def generate(self, prompts):
        outputs = self.generator.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
class vllm_generator_for_gemma2():
    def __init__(self, model,size, stops):
        self.generator = LLM(model,tensor_parallel_size=size,trust_remote_code=True,gpu_memory_utilization=0.8,enforce_eager=True,max_seq_len_to_capture=1024)
        self.sampling_params = SamplingParams(temperature=1, top_p=0.95, stop=stops, max_tokens=1024)
        
    def generate(self, prompts):
        outputs = self.generator.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]

def single_ppl(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    target_ids = inputs.clone().to(model.device)
    with torch.no_grad():
        outputs = model(inputs, labels=target_ids)
    ppl = outputs.loss.clone().detach().to("cpu")
    del inputs, target_ids,outputs
    ppl = torch.exp(ppl)
    ppl_r = ppl.numpy().item()
    return ppl_r

def get_ppl_prob(gen, sentences):
    #https://github.com/JaredFern/ImportanceSampling/blob/master/data/sample.py
    alpha = 1
    sentence_ppl = [gen.ppl([sent["instruction"]]).to("cpu") for sent in sentences]
    mean_ppl, std_ppl, ppl_99 = np.mean(sentence_ppl), np.std(sentence_ppl), np.percentile(sentence_ppl, 99)
    prob = [alpha * (p - mean_ppl) / std_ppl + 1
            if p < ppl_99 and (p - mean_ppl) / std_ppl > -1 else 1 for p in sentence_ppl]
    prob = np.array(prob)
    sentences = np.array(sentences)
    total_pr = sum(prob)
    prob[:] = [torch.tensor(p).cpu()/torch.tensor(total_pr).cpu() for p in prob]
    return prob


# def cost_time(func):
#     def fun(*args, **kwargs):
#         t = time.perf_counter()
#         result = func(*args, **kwargs)
#         print(f'func {func.__name__} cost time:{time.perf_counter() - t:.8f} s')
#         return result
#     return fun
