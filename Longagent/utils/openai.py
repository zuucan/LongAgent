import openai
import random
import time


class OpenAIKey:
    def __init__(self, keys):
        self.keys = keys
        random.seed(int(time.time()))
        self.current_key = random.choice(self.keys)

    def switch_key(self):
        if len(self.keys) == 0:
            self.current_key = None
        elif len(self.keys) == 1:
            print("No other keys available, waiting for 10s")
            time.sleep(10)
            self.current_key = self.keys[0]
        else:
            new_key = random.choice(self.keys)
            while new_key == self.current_key:
                new_key = random.choice(self.keys)
            self.current_key = new_key

    def remove_key(self):
        if self.current_key in self.keys:
            self.keys.remove(self.current_key)
        
    def process_error(self, e):
        if "RateLimitError" in repr(e) or "APIConnectionError" in repr(e) or "AuthenticationError" in repr(e):
            if "per min" in repr(e):
                print(f"Rate limit reached for key {self.current_key}")
            elif "current quota" in repr(e):
                self.remove_key()
                print(f"Remove key {self.current_key}")
                
            self.switch_key()
            if self.current_key is None:
                print("All the keys are expired!")
                exit(0)
        else:
            print(f"Unknown error: {e}")
            
            
def create_response(model="gpt-3.5-turbo", key=None, user_input=None, max_tokens=256, temperature=0.0, stop=None):
    openai.api_key = key
    client = openai.OpenAI(api_key=openai.api_key)
    if stop is None:
        response = client.completions.create(
            model=model,
            prompt=user_input,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
    else:
        response = client.completions.create(
            model=model,
            prompt=user_input,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["{}:".format(stop)]
        )
    return response["choices"][0]["text"]


def create_chat_response(model="gpt-3.5-turbo", key=None, user_input=None, max_tokens=256, temperature=0.0):
    openai.api_key = key
    client = openai.OpenAI(api_key=openai.api_key)
    SYSTEM_PROMPT = "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-3.5 architecture.\nKnowledge cutoff: 2021-09\nCurrent date: {date}"
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(date=datetime.date.today().strftime("%Y-%m-%d"))},
            {"role": "user", "content": user_input}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content


def create_multi_round_chat_response(model="gpt-3.5-turbo", key=None, response_format=None, messages=None, max_tokens=256, temperature=0.0):
    openai.api_key = key
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model=model,
        response_format=response_format,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content