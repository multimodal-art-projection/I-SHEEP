import os
import json
import argparse

def merge_json_files(sub_dir_path, output_file_path):
    # 获取子目录中的所有JSON文件并按文件名排序
    json_files = sorted([file for file in os.listdir(sub_dir_path) if file.endswith('.json')])
    
    # 初始化一个空列表来存储所有读取的数据
    merged_data = []

    # 遍历排序后的JSON文件列表
    for filename in json_files:
        file_path = os.path.join(sub_dir_path, filename)
        # 打开并读取JSON文件
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 将读取的数据添加到列表中
            merged_data.extend(data)  # 使用extend假设每个文件包含列表

    # 写入合并后的数据到指定的输出JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(merged_data, output_file, ensure_ascii=False, indent=2)
    
    print(f"Merged JSON has been written to {output_file_path}")

def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Merge JSON files in a directory into a single file.')
    parser.add_argument('--output_file', type=str, required=True, help='The path to the output JSON file.')
    
    args = parser.parse_args()


    # 获取文件所在目录
    base_directory = os.path.dirname(args.output_file)
    sub_directory = os.path.join(base_directory, 'gemma')

    # 执行合并
    merge_json_files(sub_directory, args.output_file)

if __name__ == '__main__':
    main()
