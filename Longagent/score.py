import sys
sys.path.append('../')  # 将上一级目录添加到 Python 搜索路径中
from utils.evaluate import scorer
from utils.data_loader import process_longbench
import re
import json

file_path = './logs/YOUR_OUTPUT_LOG.log'
dataset_name = "narrativeqa"
final_answers = []
ground_truths = []

# 打开文件并逐行读取内容
with open(file_path, 'r') as file:
    log_data = file.read()

# 分割成单独的条目
entries = log_data.split('\n')
# 用字典存储对应 idx 的 ground_truth
idx2ground_truth = {}

# For example, you can add your onwn dataset here.
dataset2filename = {
    "narrativeqa": "/longbench/narrativeqa_100k.jsonl"
}

ground_truth_file = dataset2filename[dataset_name]


with open(ground_truth_file, 'r', encoding='utf-8') as file:
    for idx, line in enumerate(file):
        data = json.loads(line)
        answers = data.get("answers")
        idx2ground_truth[idx] = answers
        
question_number = -1
for entry in entries:
    lines = entry.split('\n')
    
    for line in lines:
        if line.startswith('Question'):
            question_number = question_number + 1
            parts = line.split(':')  # Splitting the line based on ':'
            if len(parts) > 1:
                # question_number = parts[0].split()[1]  # Extracting the question number
                ground_truths.append(idx2ground_truth[question_number])
                print("Ground Truth:", idx2ground_truth[question_number])
        if line.startswith('Answer:'):
            final_answer = line.split(': ')[1]
            print("Answer:", final_answer)
            final_answers.append(final_answer)
        
print("Final Answers:", len(final_answers))
print("Ground Truths:", len(ground_truths))

if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
    all_classes = "null"
    score = scorer(dataset_name, final_answers, ground_truths, all_classes)
else:
    score = scorer(dataset_name, final_answers, ground_truths)
print("Score:", score)

