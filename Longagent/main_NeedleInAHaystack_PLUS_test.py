import logging
import datetime

from tqdm import tqdm
import random
from utils.openai import OpenAIKey
from utils.data_loader import process_longbench, chunk_document
from utils.evaluate import scorer

from agent import Leader, Member, Merger, Recorder

from difflib import SequenceMatcher


# 读取 API Key
gpt35_key_list = []
with open("./raw_keys.txt","r") as f:
    for line in f.readlines():
        key = line.strip()
        gpt35_key_list.append(key)

gpt35_openai_key = OpenAIKey(gpt35_key_list)
# gpt4_key_list = ['YOUR-GPT4-OPENAI-KEY']
# gpt4_openai_key = OpenAIKey(gpt4_key_list)

# For example
chunk_size = 1500
dataset_name = "needle_hotpotqa"
current_time = datetime.datetime.now()
time_string = current_time.strftime("%m-%d_%H-%M-%S")
file_path = f"/your_test_file_path.jsonl"
# file_path = f"/root/work/longbench/data_10w/test_multi_narrativeqa_100k.jsonl"
log_filename = f"logs/{dataset_name}_{chunk_size}_{time_string}.log"
logging.basicConfig(filename=log_filename, level=logging.WARNING, format='%(message)s')


STRONG_MODEL = "gpt-4-1106-preview"
WEAK_MODEL = "gpt-3.5-turbo-1106"

all_data = process_longbench(dataset_name, file_path=file_path)

all_labels = []
all_output = []


def calculate_similarity(str1, str2):
    matcher = SequenceMatcher(None, str1, str2)
    return matcher.ratio()

def remove_cluster_by_key(clusters, key_to_remove):
    # 检查指定的键是否存在于字典中
    if key_to_remove in clusters:
        del clusters[key_to_remove]

# 单轮 QA
for sample in tqdm(all_data[60:]):
    
    chunked_docs = chunk_document(sample['document'], chunk_size=chunk_size)
    num_members = len(chunked_docs)

    # leader 实例
    # leader = Leader(STRONG_MODEL, gpt4_openai_key, num_members)
    leader = Leader(WEAK_MODEL, gpt35_openai_key, num_members)
    
    # 创建 num_members 个 Member 实例
    members = []
    for member_idx in range(1, num_members + 1):
        member = Member(WEAK_MODEL, gpt35_openai_key, member_idx, chunked_docs[member_idx - 1])
        # member = Member(STRONG_MODEL, gpt4_openai_key, member_idx, chunked_docs[member_idx - 1])
        members.append(member)
    
    
    logging.warning(f"Question: {sample['task_objective']}")
    
    # 主 Agent 发出指令，子 agent 接收指令输入
    leader_response = leader.chat(sample=sample, is_first=True)
    
    # 每个子 Agent 接收指令输入，并回复
    round_count = 0
    
    pre_member_responses = ""
    first_leader_instruction = leader_response['content']
    while leader_response['type'] != "answer" and round_count < 3:
        round_count += 1
        
        now_member_responses= ""
        member_responses = ''
        clusters = {}
        for member_idx in range(num_members):
            each_member = members[member_idx]
            each_member_response = each_member.chat(leader_response['content'], pre_member_responses, first_leader_instruction)
            now_member_responses += each_member_response
            if "not contain" not in each_member_response.lower() and "no information" not in each_member_response.lower() and "not found" not in each_member_response.lower() and "not mentioned" not in each_member_response.lower() and "couldn't find" not in each_member_response.lower():
                    member_responses += "Member {}: {}".format(member_idx + 1, each_member_response)
                    # 检查当前字符串是否已经存在于字典中的某个聚类
                    found_cluster = False
                    for cluster_content, cluster_indices in clusters.items():
                        if each_member_response in cluster_content:
                            cluster_indices.append(member_idx)
                            found_cluster = True
                            break
                    if not found_cluster:
                        clusters[each_member_response] = [member_idx]
                        
        logging.warning(f"Clusters: {clusters}")
        # 处理聚类结果
        if len(clusters) >= 2:
            try_time = 0
            while len(clusters) >= 2 and try_time < 200 :
                # logging.warning(f"Clusters: {clusters}")
                try_time += 1
                first_cluster_content = list(clusters.keys())[0]
                second_cluster_content = list(clusters.keys())[1]
                first_cluster_idx_values = list(clusters.values())[0]
                second_cluster_idx_values = list(clusters.values())[1]
                first_cluster_idx =  random.choice(first_cluster_idx_values)
                second_cluster_idx = random.choice(second_cluster_idx_values)
                # merge 实例
                pre_member_responses = ''
                merger = Merger(WEAK_MODEL, gpt35_openai_key, first_cluster_idx, second_cluster_idx, chunked_docs[first_cluster_idx] + '\n' + chunked_docs[second_cluster_idx])
                merge_response = merger.chat(leader_response['content'], pre_member_responses, first_leader_instruction)
                
                if first_cluster_content != merge_response:
                    remove_cluster_by_key(clusters, first_cluster_content)
                if second_cluster_content != merge_response:
                    remove_cluster_by_key(clusters, second_cluster_content)
                if first_cluster_content != merge_response and second_cluster_content != merge_response:
                    similarity_first = calculate_similarity(first_cluster_content, merge_response)
                    similarity_second = calculate_similarity(second_cluster_content, merge_response)
                    if similarity_first > similarity_second:
                        clusters.update({merge_response: first_cluster_idx_values})
                    else:
                        clusters.update({merge_response: second_cluster_idx_values})


        logging.warning(f"final clusters: {clusters}")
        logging.warning(f"#####After verification...#####")
        new_member_responses = ''
        new_member_idx = 0
        for key in clusters.keys():
            new_member_responses += key
            # logging.warning(f"{key}")
            new_member_idx += 1

        now_member_responses = new_member_responses
        # 主 Agent 得到答案或下一条指令
        leader_response = leader.chat(sample=sample, member_responses=now_member_responses)
        pre_member_responses = now_member_responses
        
    
    # 如果是因为 round_count == 3 才跳出循环，则还需要一次专门的 Leader 轮，来给出 Answer
    if leader_response['type'] != "answer":
        leader_response = leader.chat()
    
    final_answer = leader_response['content']
    
    # 存储最终答案，方便最后评估
    logging.warning(f"Ground Truth: {sample['answer']}\n")
    all_labels.append(sample['answer'])
    all_output.append(final_answer)

# 评估
if dataset_name in ["trec", "triviaqa", "samsum", "lsht"]:
    all_classes = all_data[0]["all_classes"]
    score = scorer(dataset_name, all_output, all_labels, all_classes)
else:
    score = scorer(dataset_name, all_output, all_labels)
logging.warning(f"Score : {score}")
