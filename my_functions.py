import openai

openai.api_key = "ca392a5651064a37b2207fc766e8a3ae"
openai.api_base = "https://text-and-code-1.openai.azure.com/"
openai.api_type = 'azure'
openai.api_version = "2023-05-15"
deployment_name='gpt-35-turbo-1'

def ts_to_string(ts):
    return "Please answer following this template: (A) This city is in North America OR (B) This city is not in North America.\nMonthly average temperatures: " + str(ts)

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
    
def get_response(prompt):
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=prompt,
        temperature=0.0,
    )
    return response.choices[0]['message']['content']

def timed_get_response(prompt):
    return parse_response(get_response(prompt))
