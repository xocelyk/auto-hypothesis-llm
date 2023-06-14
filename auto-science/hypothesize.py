import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt

openai.api_key = "ca392a5651064a37b2207fc766e8a3ae"
openai.api_base = "https://text-and-code-1.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = "2023-05-15"
deployment_name='gpt-35-turbo-1'

def get_hypothesis(train_data, temperature=0.5, sample_size=16):
    system_content = '''You are WeatherBot, an AI expert in global weather patterns.  You will be given a series of monthly average temperatures for some city and some year and asked to predict if the city is in North America or not. At the end, you will be asked to provide a hypothesis for the relationship between the monthly average temperatures and whether the city is in North America or not. The goal is the generate a hypothesis that you can reference in the future to improve your predictions.'''
                        
    user_content_1 = "You will be given the average temperature for each month in Fahrenheit. The average temperatures will be given in list format. For example, if given the list [32, 45, 67, 89, 90, 87, 76, 65, 54, 43, 32, 21], the first number is the average temperature for January, the second number is the average temperature for February, and so on. You will be asked to predict if the city is in North America or not. Please answer following this template: (A) This city is in North America OR (B) This city is not in North America."
    assistant_content_1 = "Yes I understand. I am WeatherBot, and I will help identify if the city is in North America or not from its average monthly temperatures."
    user_content_2 = "Great! Let's begin :)\n"
    messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content_1}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}]
    ask_for_hypothesis = "Please provide a hypothesis for the relationship between the monthly average temperatures and whether the city is in North America or not. Please be detailed, and remember you will be using this hypothesis to improve your predictions in the future."
    messages.append({"role": "user", "content": ask_for_hypothesis})
    prompt = create_prompt(sample_size, train_data=train_data, test_data=None, messages=messages, train_mode=True)
    
    response = get_response(prompt, temperature)
    return response

