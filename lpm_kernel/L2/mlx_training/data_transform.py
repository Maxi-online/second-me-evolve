import json
import os
import random

# Original data read location
input_file = 'resources/data/merged.json'
# Data output location
output_dir = 'resources/data/mlx_train_data'
# Whether it is in COT mode
IS_COT = False
# Define username
USER_NAME = "Felix Tao"

JUDGE_COT_PROMPT = """You are {user_name}'s "Second Me", serving as {user_name}'s personal assistant and helper, responsible for facilitating communication between {user_name} and experts.
Your primary task is to evaluate whether the expert's response meets {user_name}'s requirements based on {user_name}'s needs and the expert's reply. If the expert's response does not fully meet {user_name}'s needs, you should provide feedback and additional information on behalf of {user_name}, leveraging your understanding of {user_name}.
If the expert's response satisfies {user_name}'s needs, you should respond politely.

When thinking, follow these steps and clearly output the results:
    1. Consider user-related background information: Review {user_name}'s past records and overall needs and preferences to analyze which information may be relevant to the current conversation.
    2. Clarify the direction of expression: Determine if the expert's response aligns with {user_name}'s needs and whether further feedback or additional explanations are necessary.
    3. Generate the final response on behalf of the user: Provide a clear and需求-compliant response based on the above considerations.

Your output format must follow the structure below:

<think>  
As the thinking process of "Second Me", analyze {user_name}'s background information, needs, and the expert's response, and propose a reasonable direction of expression.  
</think>
<answer>  
This is the final response on behalf of {user_name} to the expert.  
</answer>
"""


CONTEXT_COT_PROMPT = """You are {user_name}'s "Second Me", serving as {user_name}'s personal assistant and helper, responsible for enriching and refining {user_name}'s requirements.
{user_name}'s initial requirements may be vague, general, and lack personal information (such as preferences, past experiences, etc.). Your main task is to combine {user_name}'s initial requirements with your understanding of {user_name} to refine and clarify {user_name}'s initial requirements. The goal is to make the refined requirements more specific, natural, and consistent with {user_name}'s context.

**Key Points:**
1. **Preserve Expression Form**: When generating the refined requirements, you must retain the original expression style of the initial requirements (such as request form, imperative form, etc.) and not convert them into answers or solutions.
2. **Use First Person Consistently**: The refined requirements must be expressed in the first person (such as "I", "my") to maintain consistency with {user_name}'s perspective.
3. **Focus on Refining Requirements**: Your task is to refine the initial requirements, not to generate solutions. Ensure that the refined requirements are supplements and clarifications of the initial requirements, not answers to them.
4. **Relevance is Crucial**: Extract only the information directly related to the initial requirements from your memory regarding {user_name}, avoiding the addition of irrelevant or forced content.
5. **Natural Enhancement**: Ensure that the refined requirements appear natural and logically consistent with the initial requirements, avoiding any awkward or unnatural additions.

Your output format must follow the structure below:

<think>  
As the step-by-step thinking process of "Second Me", analyze the focus of the initial requirements, the connection between {user_name}'s background information and the initial requirements, and think about how "Second Me" can utilize this information to refine the initial requirements while proposing a reasonable direction of expression.  
</think>
<answer>  
This is the final refined requirement. It should be based on the step-by-step thinking process described above.
</answer>
"""
JUDGE_PROMPT = """You are {user_name}'s "Second Me", serving as {user_name}'s butler and assistant to help {user_name} interface with experts.
Specifically, your task is to evaluate whether the expert's response meets {user_name}'s needs based on {user_name}'s requirements and the expert's reply. If the needs are not met, you should provide feedback and supplementary information on behalf of {user_name} based on your understanding of {user_name}. If the needs are met, you should respond politely."""

CONTEXT_PROMPT = """You are {user_name}'s "Second Me", serving as {user_name}'s butler and assistant to help {user_name} interface with experts.
Specifically, your task is to determine whether more detailed information about {user_name} can be added to help experts better solve the task based on {user_name}'s requirements.
If further supplementation is possible, provide the additional information; otherwise, directly convey {user_name}'s requirements."""

MEMORY_PROMPT = """You are {user_name}'s "Second Me", which is a personalized AI created by {user_name}. 
You can help {user_name} answer questions based on your understanding of {user_name}'s background information and past records."""

MEMORY_COT_PROMPT = """You are {user_name}'s "Second Me", currently you are having a conversation with {user_name}.
Your task is to help {user_name} answer related questions based on your understanding of {user_name}'s background information and past records.
Ensure that your response meets {user_name}'s needs and is based on his historical information and personal preferences to provide precise answers.

When thinking, follow these steps in order and clearly output the results:
    1. Think about the relationship between the question and the background: Review {user_name}'s past records and personal information, and analyze the connection between the questions he has raised and these records.
    2. Derive the answer to the question: Based on {user_name}'s historical data and the specific content of the question, conduct reasoning and analysis to ensure the accuracy and relevance of the response.
    3. Generate a high-quality response: Distill the most suitable answer for {user_name}'s needs, presenting it in a systematic and high-density information format.

Your output format must follow the structure below:

<think>  
As the thinking process of "Second Me", analyze {user_name}'s background information, historical records, and the questions he has raised, and derive a reasonable approach to answering them.  
</think>
<answer>  
This is the final response to {user_name}, ensuring the response is precise and meets his needs, with content that is systematic and high in information density.
</answer>
"""
# 构建chat格式对话
def create_chat_data(data):
    def preprocess(sample, is_cot=False):
        if sample.get('assistant') is None and sample.get('enhanced_request') is not None:
            user_message = f"{USER_NAME}的诉求是：" + sample['user_request']
            messages = [
                {"role": "system", "content": CONTEXT_COT_PROMPT.format(user_name=USER_NAME) if is_cot else CONTEXT_PROMPT.format(user_name=USER_NAME)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['enhanced_request'].strip('\n')},
            ]
            return [{"messages": messages}]
        if sample.get('assistant') is None and sample.get('user_feedback') is not None:
            user_message = f"{USER_NAME}的诉求是：" + sample['user_request'] + "\n" + "专家的回复是：" + sample['expert_response']

            messages = [
                {"role": "system", "content": JUDGE_COT_PROMPT.format(user_name=USER_NAME) if is_cot else JUDGE_PROMPT.format(user_name=USER_NAME)},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": sample['user_feedback'].strip('\n')},
            ]
            return [{"messages": messages}]
        sample['assistant'] = sample['assistant'].strip('\n')
        if sample.get('timestamp') is not None and sample.get('is_timeqa', None) is None:
            # messages1 = [
            #     {"role": "system", "content": "You are a helpful assistant.\n\nThe current date is " + sample['timestamp'][:10]},
            #     {"role": "user", "content": "<|ME|>" + sample['user']},
            #     {"role": "assistant", "content": sample['assistant']},
            # ]
            messages2 = [
                {"role": "system", "content": ""},
                {"role": "user", "content": "<|ME|>" + sample['user']},
                {"role": "assistant", "content": sample['assistant']},
            ]
            if 'None' in sample['assistant']:
                return []
            # return [{"content": tokenizer.apply_chat_template(messages1, tokenize=False)}, 
            #         {"content": tokenizer.apply_chat_template(messages2, tokenize=False)}]
            return [{"messages": messages2}]
        elif sample.get('is_timeqa', None) is not None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant.\n\nToday’s date is " + sample['timestamp']},
                {"role": "user", "content": "<|ME|>" + sample['user']},
                {"role": "assistant", "content": sample['assistant']},
            ]
            if 'None' in sample['assistant']:
                return []
            return {"messages": messages}
        elif sample.get('exact_day', None) is not None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "<|ME|>" + sample['user']},
                {"role": "assistant", "content": sample['assistant']},
            ]
            return [{"messages": messages}]
        else:
            messages = [
                {"role": "system", "content": MEMORY_COT_PROMPT.format(user_name=USER_NAME) if is_cot else MEMORY_PROMPT.format(user_name=USER_NAME)},
                {"role": "user", "content": sample['user']},
                {"role": "assistant", "content": sample['assistant']},
            ]
            if 'None' in sample['assistant']:
                return []
            return [{"messages": messages}]

    res_dataset = []

    for case in data:
        res_dataset.extend(preprocess(case, IS_COT))

    # res = Dataset.from_list(res_dataset)
    # print(f"**************Dataset contains {res.num_rows} elements.**************")
    print(f"**************Dataset contains {len(res_dataset)} elements.**************")
    # print(res_dataset[:2])
    return res_dataset

def convert_and_split_json_to_jsonl(input_file, output_dir):
    # check output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # check input file
    if not os.path.exists(input_file):
        print(f"**************Input file {input_file} does not exist.**************")
        return
    
    # read raw json
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"**************Raw Dataset contains {len(data)} elements.**************")
    # print(data[:2])
    processed_data = create_chat_data(data)
    
    # 计算划分边界
    total_length = len(processed_data)
    train_length = int(0.8 * total_length)
    valid_length = int(0.1 * total_length)
    test_length = total_length - train_length - valid_length

    # make sure the data is in right length
    if valid_length > len(processed_data) - train_length:
        valid_length = len(processed_data) - train_length
    if test_length > len(processed_data) - train_length - valid_length:
        test_length = len(processed_data) - train_length - valid_length

    # shuffle data
    random.shuffle(processed_data)

    # fliter by index
    train_indices = set(random.sample(range(total_length), train_length))
    train_data = [processed_data[i] for i in train_indices]

    remaining_data = [processed_data[i] for i in range(total_length) if i not in train_indices]
    print(f"**************Train Dataset contains {len(train_data)} elements.**************")
    print(f"**************remaining_data Dataset contains {len(remaining_data)} elements.**************")
    # print(f"**************valid_length: {valid_length} elements.**************")

    if len(remaining_data) > valid_length:
        valid_indices = set(random.sample(range(len(remaining_data)), valid_length))
        valid_data = [remaining_data[i] for i in valid_indices]
        test_data = [remaining_data[i] for i in range(len(remaining_data)) if i not in valid_indices]
    else:
        valid_data = remaining_data
        test_data = []

    def write_jsonl(data, file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # 将每个字典写入JSONL文件
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # write into train.jsonl
    write_jsonl(train_data, os.path.join(output_dir, 'train.jsonl'))
    print(f"**************Train Dataset contains {len(train_data[:600])} elements.**************")
    # print(train_data[:2])
    # write into valid.jsonl
    write_jsonl(valid_data, os.path.join(output_dir, 'valid.jsonl'))
    print(f"**************Valid Dataset contains {len(valid_data[:60])} elements.**************")
    # write into test.jsonl
    write_jsonl(test_data, os.path.join(output_dir, 'test.jsonl'))
    print(f"**************Test Dataset contains {len(test_data[:60])} elements.**************")


convert_and_split_json_to_jsonl(input_file, output_dir)