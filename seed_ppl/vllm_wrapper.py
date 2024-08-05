from transformers import AutoModelForCausalLM, AutoTokenizer
import json
device = "cuda" # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained(
    "/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B",
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("/ML-A100/team/mm/eamon/self_instruction/seed_ppl/models/Qwen_72B", trust_remote_code=True)
tokenizer.eos_token = "<|endoftext|>" 
tokenizer.pad_token = tokenizer.eos_token
# prompt = "Give me a short introduction to large language model."
while True:
    prompt = input()
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(text)
    with open("input.jsonl", "w", encoding='utf-8') as f:
        json.dump(text, f, ensure_ascii=False)
        
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)