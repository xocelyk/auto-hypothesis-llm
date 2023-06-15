import json

def load_prompts():
    with open('prompt_strings.json', 'r') as f:
        prompts = json.load(f)
    return prompts
