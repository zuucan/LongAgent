import json
import tiktoken


dataset2description = {
    "narrativeqa": "There is the text of a very long story that is a book or movie script. You need to answer a question based on the content of the story. The answer to the question can always be found in the text, and you should provide a direct response to the question asked.",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write 'unanswerable'. If the question is a yes/no question, answer 'yes', 'no', or 'unanswerable'. Do not provide any explanation.",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "trec": "This is a question classification task. You will receive many 'Question-Type' samples, and a question to be classified. You need to learn how to classify from the samples and determine the type of the given question. No need to come up with an answer to the question, just predict the type. A recommended strategy is to have each member retrieve and return samples relevant to the given question, and then you make the final decision.\nHere are some examples of input-output pair:\nQuestion: What operating system do IBM-compatible machines use?\nType: Produc\nQuestion: What state is John F. Kennedy buried in?\nType: State\nQuestion: What Marx Brothers movie centers on a stolen painting?\nType: Invention, book and other creative piece\nQuestion: What is `Last Chance for Animals'?\nType: Description of something\nQuestion: How many sides does a heptagon have?\nType: Number of something",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.",
    "needle": "Find the answer to the question in the document, there is a lot of distracting information in the documentation.",
    "squad": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "needle_squad": "Answer the question based on the given passages. The answer must be extracted from the given passages.",
    "needle_hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
}

# 每个样例有：task_description、task_objective、完整 document、answer
def process_longbench(dataset_name,file_path):
    if file_path is None:
        file_path = f"/root/work/longbench/data/{dataset_name}.jsonl"
    extracted_data_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            answers = data.get("answers")
            document = data.get("context")   
            task_objective = data.get("input")
            if dataset_name == "gov_report":
                task_objective = "Write a one-page summary of the government report."
            all_classes = data.get("all_classes")
            task_description = dataset2description[dataset_name]
            
            extracted_data = {
                "task_description": task_description,
                "task_objective": task_objective,
                "document": document,
                "answer": answers,
                "all_classes": all_classes
            }

            extracted_data_list.append(extracted_data)

    return extracted_data_list

# 每个样例有：task_description、task_objective、完整 document、answer
def process_file(file_path):
    extracted_data_list = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            prompt = data.get("prompt")
            answer = data.get("expected_number")
            task_name = data["random_idx"][0]

            # Format task objective
            template = "What is the <REGISTER_CONTENT> in line {}? I need the number."
            task_objective = template.format(task_name)

            task_description = "There is a record of lines. Each line begins with 'line <line index>' and contains a '<REGISTER_CONTENT>' at the end of the line as a numerical value. For each line index, memorize its corresponding <REGISTER_CONTENT>. At the end of the record, I will ask you to retrieve the corresponding <REGISTER_CONTENT> of a certain line index.\nHere is an example of one of these lines:\nline grotesque-classmate: REGISTER_CONTENT is <42527>"

            start_prefix = "Now the record start:\n\n"
            end_prefix = "\n\nNow the record is over."
            if prompt and start_prefix in prompt:
                start_index = prompt.index(start_prefix) + len(start_prefix)
                end_index = prompt.index(end_prefix)
                document = prompt[start_index:end_index]

                extracted_data = {
                    "task_description": task_description,
                    "task_objective": task_objective,
                    "document": document,
                    "answer": answer
                }

                extracted_data_list.append(extracted_data)

    return extracted_data_list

# 提取指定行的 task_objective、完整 document、answer，写入到一个txt文件中
def extract_content(file_path):
    row_list = [3, 9, 28, 30]
    extracted_data_list = []
    process_file(file_path)
    for row in row_list:
        extracted_data_list.append(process_file(file_path)[row])
    with open("wrong.txt", "w", encoding='utf-8') as f:
        for extracted_data in extracted_data_list:
            f.write("task objective:\n" + extracted_data["task_objective"] + "\n")
            f.write("document:\n" + extracted_data["document"] + "\n")
            f.write("answer:\n" + str(extracted_data["answer"]) + "\n\n")
            


def chunk_document(document, chunk_size):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(document)
    chunked_tokens = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    chunked_docs = [encoding.decode(chunk) for chunk in chunked_tokens]
    return chunked_docs