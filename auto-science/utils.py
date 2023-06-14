import numpy as np
import pandas as pd
import numpy as np
import openai
from data_loader import load_data

openai.api_key = "ca392a5651064a37b2207fc766e8a3ae"
openai.api_base = "https://text-and-code-1.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = "2023-05-15"
deployment_name='gpt-35-turbo-1'

def get_train_ts_label(sample_size, train_data):
    ts_list = []
    label_list = []
    train_keys = list(train_data.keys())
    np.random.shuffle(train_keys)
    train_keys = train_keys[:sample_size]
    for key in train_keys:
        sample_ts = train_data[key]['MonthlyAvgTemperature']
        sample_label = train_data[key]['Label']
        ts_list.append(sample_ts)
        label_list.append(sample_label)
    return ts_list, label_list

def create_prompt(num_shots, train_data=None, test_data=None, messages=[], train_mode=False, test_mode=False):
    # train and test not mutually exclusive
    # if only train is on, we do not include the test prompt
    # if only test is on, we do not include the train prompt (zero-shot)

    if train_mode:
        train_ts_list, train_label_list = get_train_ts_label(num_shots, train_data)
        for i in range(num_shots):
            messages.append({"role": "user", "content": ts_to_string(train_ts_list[i])})
            messages.append({"role": "assistant", "content": label_to_string(train_label_list[i])})
    
    if test_mode:
        test_ts, test_label = get_test_ts_label(test_data)
        messages.append({"role": "user", "content": ts_to_string(test_ts)})

    return messages

def get_test_ts_label(test_data):
    return test_data['MonthlyAvgTemperature'], test_data['Label']

def ts_to_string(ts):
    return "Monthly average temperatures: " + str(ts)

def label_to_string(label):
    if label == 1:
        return "(A) This city is in North America."
    else:
        return "(B) This city is not in North America."

def parse_response(response_string):
    # return 1 if correct, 0 if incorrect, -1 if invalid response
    if ('(A)' in response_string and '(B)' in response_string) or ('(A)' not in response_string and '(B)' not in response_string): # invalid response
        return -1
    else:
        return int('(A)' in response_string)
    
def get_response(prompt, temperature=0.5):
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=prompt,
        temperature=temperature,
    )
    return response.choices[0]['message']['content']

