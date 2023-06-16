import json

def load_prompts(filename='prompts/weather.json'):
    with open(filename, 'r') as f:
        prompts = json.load(f)
    return prompts
