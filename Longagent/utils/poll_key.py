import openai
from utils.openai import OpenAIKey
from utils.openai import create_response, create_chat_response
from tqdm import tqdm

raw_key_list = []

with open("./raw_keys.txt","r") as f:
    for line in f.readlines():
        key = line.strip()
        raw_key_list.append(key)

openai_key = OpenAIKey(raw_key_list)


# # test once
response = create_chat_response(
    model="gpt-3.5-turbo",
    user_input="Hello!",
    max_tokens=32,
    temperature=0
)
print(response)

# # test 5 times
# success = False  # 添加一个标志以跟踪是否成功
# for _ in tqdm(range(5)):
#     if success:  # 如果成功，跳出循环
#         break
    
#     try_times = 0
#     while try_times < 5:
#         try: 
#             response = create_chat_response(
#                 model="gpt-3.5-turbo",
#                 user_input="Hello!",
#                 max_tokens=32,
#                 temperature=0
#             )
#             print(response)
#             success = True  # 标记成功
#             break
#         except Exception as e:
#             try_times += 1
#             if try_times == 5:
#                 print("Try 5 times, but failed! Skip this one.")
                
#             if "RateLimitError" in repr(e) or "APIConnectionError" in repr(e) or "AuthenticationError" in repr(e):
#                 if "per min" in repr(e):
#                     print(f"Rate limit reached for key {openai.api_key}")
#                 elif "current quota" in repr(e):
#                     openai_key.remove_key()
#                     print(f"Remove key {openai.api_key}")
#                 if openai_key.switch_key() is None:
#                     print("All the keys are expired!")
#                     exit(0)
#             else:
#                 print(f"Unknown error: {e}")