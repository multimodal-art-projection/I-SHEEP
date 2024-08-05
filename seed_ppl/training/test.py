import json

# Read the .jsonl file and parse each line
with open('./all_instances_82K.jsonl', 'r') as jsonl_file:
    json_list = []
    for index, line in enumerate(jsonl_file):
        json_obj = json.loads(line)
        json_obj['example_id'] = index
        json_list.append(json_obj)

# Write the parsed JSON objects to a .json file
with open('./all_data.json', 'w') as json_file:
    json.dump(json_list, json_file, indent=4)
# Write the parsed JSON objects to a .json file
with open('./all_data_debug.json', 'w') as json_file:
    json.dump(json_list[:3000], json_file, indent=4)

