##########################################
################# Leader #################
##########################################

Leader_Start_Template = """# Task Description
{task_description}

# Task Objective
{task_objective}

# Generate Instruction for Members
Now, you need to generate an instruction for all team members. You can ask them to answer a certain question, or to extract information related to the task, based on their respective documents.
Your output must following the JSON format: {{"type": "instruction", "content": "your_instruction_content"}}"""


Leader_Next_Template = """Here are the responses from all the members. Each member sees different segments of a document, and these segments do not intersect with each other. The correct answer may appear in any one or several members' responses.
Note that if a minority of members find information relevant to the question while the majority reply that the document does not contain information relevant to the question, you should pay attention to the replies from those members who found relevant information.

# Member Response
{member_response}

# Task Description
{task_description}

# Task Objective
{task_objective}

# Determination
Based on the above information, you need to determine if you can solve the task objective. You have two choices:
1. If members' responses cannot solve the task objective, or if their responses contain conflicting answers, provide a new instruction for them to answer again.
2. Else, if the task objective can be solved, give your final answer as concisely as you can, using a single phrase if possible. Do not provide any explanation.
Your output must following the JSON format: {{"type": "answer", "content": "your_answer_content"}} or {{"type": "instruction", "content": "your_instruction_content"}}"""


Leader_End_Template = """You have reached the maximum number of conversation rounds, please ignore the last instruction.

# Generate Final Answer
Now, you need to generate final answer to achieve the task objective.
Your output must following the JSON format: {{"type": "answer", "content": "your_answer_content"}}"""


##########################################
################# Member #################
##########################################

Member_Start_Template = """# Document
{member_document}

# Instruction
{leader_instruction}
You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write “unanswerable”. If the question is a yes/no question, answer “yes”, “no”, or “unanswerable”. Do not provide any explanation.
Your output must following the JSON format: {{"type": "response", "content": "your_response_content"}}
The "content" needs to be as concise as possible.
"""


Member_Next_Template = """# Document
{member_document}
Great response! However, based on all member responses from the last round, there is not yet able to get a final answer. Here comes a new instruction.

# Member Response Last Round
{member_responses}

# New Instruction
{leader_instruction}

Answer the question based on the given document. Only give me the answer and do not output any other words.
Remember, your output must following the JSON format: {{"type": "response", "content": "your_response_content"}}
"""