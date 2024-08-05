import json
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default='', type=str)
parser.add_argument("--output_file", default='', type=str)
parser.add_argument("--threshold", default='', type=str)

args = parser.parse_args()

threshold=int(args.threshold)

def filter_by_prompt_eval_score(data):
    print("filtering by prompt eval score...")
    print(f"origin data len is: {len(data)}")
    # quality_score >= 7 and following_score >= 7
    maintain_data = [entry for entry in data if entry['quality_score']>threshold and entry['following_score']>threshold]
    # maintain_data = [entry for entry in maintain_data if entry['following_score']> 6]
    
    filtered_data = [entry for entry in data if entry['quality_score']<threshold+1 or entry['following_score']<threshold+1]
    # filtered_data = [entry for entry in filtered_data if entry['following_score']<7]
    
    print(f"the maintain data len is: {len(maintain_data)}")
    print(f"the filtered data len is: {len(data) - len(maintain_data)}")
    return maintain_data, filtered_data


def save_json(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        # for entry in data:
        #     json.dump(entry, file)
        #     file.write('\n')

def save_jsonl(data, output_path):
    """Saves a list of dictionaries to a JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as file:
        # json.dump(data, file, ensure_ascii=False, indent=2)
        for entry in data:
            json.dump(entry, file, ensure_ascii=False)
            file.write('\n')

def main(data_path, targetfp):
    # with open(data_path, "r", encoding='utf-8') as f:
    #     total_data = [json.loads(l) for l in f]
    with open(data_path, "r", encoding='utf-8') as f:
        total_data = json.load(f)
        # total_data = total_data[:30] + total_data[len(total_data)//2:len(total_data)//2+30] + total_data[-30:]
        
    maintain_data, filtered_data = filter_by_prompt_eval_score(total_data)
    save_json(maintain_data, targetfp)
    save_jsonl(filtered_data, targetfp.replace('.json', '_look_filtered.jsonl'))
    

# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/look_test_filter_by_form_score.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/look_test_filter_by_form_score_filter_by_eval_score.json')
    
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_2epoch_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter3_2epoch_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter3_2epoch_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_on_base_2epoch_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_on_base_2epoch_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_one_step_2epoch_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_10k_iter2_one_step_2epoch_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_20k_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_20k_output_filtered_by_promt_eval.json')
# main('/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_30k_output_filtered_by_promt_eval_origin.json', '/ML-A100/team/mm/eamon/self_instruction/seed_ppl/outputs/seed_ppl_30k_output_filtered_by_promt_eval.json')

main(args.input_file, args.output_file)