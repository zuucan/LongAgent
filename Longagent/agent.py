import json
import logging
import datetime
import re
from utils.openai import create_chat_response, create_multi_round_chat_response


from prompt_template import Leader_Start_Template, Leader_Next_Template, Leader_End_Template,\
                            Member_Start_Template, Member_Next_Template


GPT35_SYSTEM_PROMPT = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: {date}"
GPT4_SYSTEM_PROMPT = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.\nKnowledge cutoff: 2023-04\nCurrent date: {date}"

LEADER_SYSTEM_PROMPT = """You are the leader of a team of {num_members} members. Your team will need to collaborate to solve a task. The rule is:
1. Only you know the task description and task objective; the other members do not.
2. But they will receive different documents that may contain answers, and you need to send them an instruction to query their document.
3. Your instruction need to include your understanding of the task and what you need them to focus on. If necessary, your instructions can explicitly include the task objective.
4. Finally, you need to complete the task based on the query results they return."""

MEMBER_SYSTEM_PROMPT = """You are a member of a team. Your team are collaborating to solve a task. 
# Each member will receive a different document and an instruction. You need to respond according to your document so that the team can make further decisions.
"""

class Agent:
    def __init__(self, model, openai_key):
        self.model = model
        self.openai_key = openai_key
        self.max_try_times = 20
        
        self.messages = []
        
        
    def generate_response(self, max_tokens=512, temperature=0):
        try_times = 0
        response = ""
        while try_times < self.max_try_times:
            try:
                # self.openai_key.switch_key()
                response = create_multi_round_chat_response(
                    model=self.model,
                    key=self.openai_key.current_key,
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                break
            
            except Exception as e:
                try_times += 1
                if try_times == self.max_try_times:
                    logging.warning(f"Try {self.max_try_times} times, but failed! Skip this one.")
                self.openai_key.process_error(e)
        
        return response

    def generate_json_response(self, max_tokens=512, temperature=0):
        try_times = 0
        json_response = {"type": "error", "content": "Generate response error!"}
        while try_times < self.max_try_times:
            try:
                # 如果要并行则必须有这句话
                # self.openai_key.switch_key()
                # logging.warning("!!!!!!!content of message:{}".format(self.messages))
                json_string = create_multi_round_chat_response(
                    model=self.model,
                    key=self.openai_key.current_key,
                    response_format={"type": "json_object"},
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                json_response = json.loads(json_string)
                break
            
            except Exception as e:
                try_times += 1
                if try_times == self.max_try_times:
                    logging.warning(f"Try {self.max_try_times} times, but failed! Skip this one.")
                self.openai_key.process_error(e)

        return json_response
        
    def update_messages(self, role, user_input):
        self.messages.append({"role": role, "content": user_input})
        
    def update_member_messages(self, role, user_input):
        for message in self.messages:
            input_string = message['content']
            # 定义正则表达式模式
            pattern1 = re.compile(r'# Document(.*?)# Member Response Last Round', re.DOTALL)
            pattern2 = re.compile(r'# Document(.*?)# Instruction', re.DOTALL)

            # 判断是否匹配到模式1
            match1 = pattern1.search(input_string)
            if match1:
                # 将匹配到的部分替换为指定的文本
                replaced_content = 'The specific document content is omitted here'
                message['content'] = pattern1.sub(replaced_content, input_string)

            # 判断是否匹配到模式2
            match2 = pattern2.search(input_string)
            if match2:
                # 将匹配到的部分替换为指定的文本
                replaced_content = 'The specific document content is omitted here'
                message['content'] = pattern2.sub(replaced_content, input_string)
                
        self.messages.append({"role": role, "content": user_input})
        
class Leader(Agent):
    def __init__(self, model, openai_key, num_members):
        super().__init__(model, openai_key)
        self.num_members = num_members
        self.messages.append({"role": "system", "content": LEADER_SYSTEM_PROMPT.format(num_members=num_members)})


    def load_start_prompt(self, sample):
        leader_start_prompt = Leader_Start_Template.format(
            # num_members=self.num_members,
            task_description=sample['task_description'],
            task_objective=sample['task_objective']
        )
        self.update_messages("user", leader_start_prompt)

    def load_next_prompt(self, member_responses, sample):
        leader_next_prompt = Leader_Next_Template.format(
            member_response=member_responses,
            task_description=sample['task_description'],
            task_objective=sample['task_objective']
        )
        self.update_messages("user", leader_next_prompt)
        
    def load_end_prompt(self, member_responses):
        leader_end_prompt = Leader_End_Template.format(
            member_response=member_responses,
        )
        self.update_messages("user", leader_end_prompt)

    def chat(self, sample=None, member_responses=None, is_first=False):
        if is_first:
            self.load_start_prompt(sample)
        elif member_responses != None:
            self.load_next_prompt(member_responses,sample)
        else:
            self.load_end_prompt(member_responses)
        
        while True:
            json_response = self.generate_json_response()
            
            if "instruction" in json_response['type'].lower():
                logging.warning(f"Instruction: {json_response['content']}")
                break
            elif "answer" in json_response['type'].lower():
                logging.warning(f"Answer: {json_response['content']}")
                break
            else:
                continue
            
        self.update_messages("assistant", str(json_response))
        
        return json_response


class Member(Agent):
    def __init__(self, model, openai_key, member_idx, document_chunk):
        super().__init__(model, openai_key)
        self.member_idx = member_idx
        self.document_chunk = document_chunk
        self.messages.append({"role": "system", "content": MEMBER_SYSTEM_PROMPT})
        
    def load_start_prompt(self, leader_instruction):
        member_start_prompt = Member_Start_Template.format(
            leader_instruction=leader_instruction, 
            member_document=self.document_chunk
        )
        self.update_messages("user", member_start_prompt)
        
    def load_next_prompt(self, leader_instruction, member_responses):
        member_next_prompt = Member_Next_Template.format(
            member_document=self.document_chunk,
            leader_instruction=leader_instruction, 
            member_responses=member_responses
        )
        self.update_member_messages("user", member_next_prompt)

    def chat(self, leader_instruction, member_responses,first_leader_instruction):
        if member_responses == "":
            self.load_start_prompt(leader_instruction)
        else:
            self.load_next_prompt(leader_instruction, member_responses)

        while True:
            json_response = self.generate_json_response()
            
            if "response" in json_response['type'].lower():
                logging.warning(f"Member {self.member_idx}: {json_response['content']}")
                break
            else:
                continue
                
        self.update_messages("assistant", str(json_response))
        
        response_str = f"Member {self.member_idx}: {json_response['content']}\n"
        return response_str
 

class Merger(Agent):
    def __init__(self, model, openai_key, member_idx_first, member_idx_second, document_chunk):
        super().__init__(model, openai_key)
        self.member_idx_first = member_idx_first
        self.member_idx_second = member_idx_second
        self.document_chunk = document_chunk
        self.messages.append({"role": "system", "content": MEMBER_SYSTEM_PROMPT})
        
    def load_start_prompt(self, leader_instruction):
        member_start_prompt = Member_Start_Template.format(
            leader_instruction=leader_instruction, 
            member_document=self.document_chunk
        )
        self.update_messages("user", member_start_prompt)
        
    def load_next_prompt(self, leader_instruction, member_responses):
        member_next_prompt = Member_Next_Template.format(
            member_document=self.document_chunk,
            leader_instruction=leader_instruction, 
            member_responses=member_responses
        )
        self.update_member_messages("user", member_next_prompt)

    def chat(self, leader_instruction, member_responses,first_leader_instruction):
        if member_responses == "":
            self.load_start_prompt(leader_instruction)
        else:
            self.load_next_prompt(leader_instruction, member_responses)

        while True:
            json_response = self.generate_json_response()
            
            if "response" in json_response['type'].lower():
                logging.warning(f"Member {self.member_idx_first} and Member {self.member_idx_second} merged: {json_response['content']}")
                break
            else:
                continue
                
        self.update_messages("assistant", str(json_response))
        
        response_str = f"Member {self.member_idx_first}: {json_response['content']}\n"
        return response_str
    
