import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt
from dotenv import load_dotenv
import os
from load_prompts import load_prompts

prompts = load_prompts()

load_dotenv()

api_key = os.getenv('API_KEY')
api_base = os.getenv('API_BASE')
api_type = os.getenv('API_TYPE')
api_version = os.getenv('API_VERSION')
deployment_name = os.getenv('DEPLOYMENT_NAME')

openai.api_key = api_key
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version
deployment_name = deployment_name

def get_hypothesis(train_data, temperature=0.5, sample_size=16):
    system_content = prompts['SYSTEM_CONTENT_1']
    user_content_1 = prompts['USER_CONTENT_1']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    ask_for_hypothesis = prompts['ASK_FOR_HYPOTHESIS']
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content_1}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}, {"role": "user", "content": ask_for_hypothesis}]
    prompt = create_prompt(sample_size, train_data=train_data, test_data=None, messages=messages, train_mode=True)
    response = get_response(prompt, temperature)
    return response

