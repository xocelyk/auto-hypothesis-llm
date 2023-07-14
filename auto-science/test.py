import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt, get_test_ts_label, parse_response
from dotenv import load_dotenv
import os
from config import load_config
import time

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

config = load_config()
prompts = config['prompts']
data_mode = config['data_mode']
TIMEOUT = 5
# TODO: this shouldn't be global

import multiprocessing

def worker_task(arg):
    return get_response_with_timeout(arg, 0.0)

def get_response_with_timeout(prompt, temperature):
    return get_response(prompt, temperature)

def test_hypothesis_one_example(hypothesis, num_shots=0, test_icl_data=None, test_validation_data=None):
    system_content = prompts['SYSTEM_CONTENT_2']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    user_content_3 = prompts['USER_CONTENT_3']
    hypothesis_prompt = 'Please use the hypothesis to help you with your predictions. Here is the hypothesis: ' + hypothesis
    # take first item from dictionary
    test_sample = test_validation_data
    test_ts, test_label = get_test_ts_label(test_sample)
    messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}, {"role": "user", "content": hypothesis_prompt}]
    prompt = create_prompt(num_shots=num_shots, train_data=test_icl_data, test_data=test_sample, train_mode=True, test_mode=True, messages=messages)
    prompt.append({"role": "user", "content": user_content_3})
    remind_hypothesis_text = 'Please use the hypothesis to make your prediction, explain your reasoning, and follow the answer template.'
    prompt.append({"role": "user", "content": remind_hypothesis_text})
    response_text = get_response(prompt, temperature=0.7, timeout=TIMEOUT)[0]
    response = parse_response(response_text)
    if response == test_label:
        return 1
    else:
        if response == -1:
            return -1
        else:
            return 0


def test_hypothesis(hypothesis, num_shots=0, test_icl_data=None, test_validation_data=None, verbose=True):
    # set up for stat collection
    first = True
    correct = incorrect = invalid = api_timeout = total = 0

    # prompt set up
    system_content = prompts['SYSTEM_CONTENT_2']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    user_content_3 = prompts['USER_CONTENT_3']
    hypothesis_prompt = 'Please use the hypothesis to help you with your predictions. Here is the hypothesis: ' + hypothesis

    # experiment
    test_keys = list(test_validation_data.keys())
    keys_to_retry = []
    responses = []
    gts = []
    with multiprocessing.Pool(processes=1) as pool:
        for key in test_keys:
            time.sleep(1)
            test_sample = test_validation_data[key]
            test_ts, test_label = get_test_ts_label(test_sample)
            messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}, {"role": "user", "content": hypothesis_prompt}]
            prompt = create_prompt(num_shots=num_shots, train_data=test_icl_data, test_data=test_sample, train_mode=True, test_mode=True, messages=messages)
            prompt.append({"role": "user", "content": user_content_3})
            remind_hypothesis_text = 'Please use the hypothesis to make your prediction, explain your reasoning, and follow the answer template.'
            prompt.append({"role": "user", "content": remind_hypothesis_text})

            # if first: # write prompt to text file
            #     with open(f'data/generated_prompts/test_prompt_{data_mode}.txt', 'w') as f:
            #         for el in prompt:
            #             f.write(el['role'] + ': ' + el['content'] + '\n\n')

            try:
                result = pool.apply_async(get_response_with_timeout, args=(prompt, 0.0))
                response_text = result.get(timeout=TIMEOUT)[0]
            except multiprocessing.TimeoutError:
                print(f'API call exceeded {TIMEOUT} seconds')
                keys_to_retry.append(key)
                continue
            except openai.error.APIConnectionError:
                print('API Connection Error')
                keys_to_retry.append(key)
                continue
            except OSError as e:
                if e.errno == 24:
                    keys_to_retry.append(key)
                    print('Too many open files')
                    continue  # go to the next loop iteration
                else:
                    raise e # wait for the pool to close by joining
            response = parse_response(response_text)
            print(messages)
            print(response_text)
            responses.append(response)
            gts.append(test_label)
            if response == -1:
                print(response_text)
                print()
                invalid += 1
            elif response == test_label:
                correct += 1
            else:
                incorrect += 1
            total += 1

            # descriptive stats
            tp = np.array([1 if response == 1 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
            fp = np.array([1 if response == 1 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
            tn = np.array([1 if response == 0 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
            fn = np.array([1 if response == 0 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
            recall = tp/(tp + fn)
            precision = tp/(tp + fp)
            f1 = 2 * (precision * recall) / (precision + recall)

            if verbose:
                try: # handle divide by zero
                    print('Details: ' + str({
                        'Accuracy': round(correct/(incorrect + correct), 3), 
                        'Correct': correct, 
                        'Incorrect': incorrect, 
                        'Invalid': invalid, 
                        'API Timeout': api_timeout, 
                        'Total': total, 
                        'TP': tp, 
                        'FP': fp, 
                        'TN': tn, 
                        'FN': fn, 
                        'F1': round(f1, 3), 
                        'Recall': round(recall, 3), 
                        'Precision': round(precision, 3)
                    }))
                except:
                    print('Accuracy:', 0, 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'API Timeout:', api_timeout, 'Total:', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', 0, 'Recall:', 0, 'Precision:', 0)
        for key in keys_to_retry:
            test_sample = test_validation_data[key]
            test_ts, test_label = get_test_ts_label(test_sample)
            messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}, {"role": "user", "content": hypothesis_prompt}]
            prompt = create_prompt(num_shots=num_shots, train_data=test_icl_data, test_data=test_sample, train_mode=True, test_mode=True, messages=messages)
            prompt.append({"role": "user", "content": user_content_3})
            remind_hypothesis_text = 'Please use the hypothesis to make your prediction, explain your reasoning, and follow the answer template.'
            prompt.append({"role": "user", "content": remind_hypothesis_text})

            try:
                result = pool.apply_async(get_response_with_timeout, args=(prompt, 0.0))
                response_text = result.get(timeout=TIMEOUT)[0]
            except multiprocessing.TimeoutError:
                print(f'API call exceeded {TIMEOUT} seconds')
                api_timeout += 1
                continue
            except openai.error.APIConnectionError:
                print('API Connection Error')
                api_timeout += 1
                continue
            except OSError as e:
                if e.errno == 24:
                    print('Too many open files')
                    api_timeout += 1
                    continue  # go to the next loop iteration
                else:
                    raise e # wait for the pool to close by joining
            response = parse_response(response_text)
            responses.append(response)
            gts.append(test_label)
            if response == -1:
                print(response_text)
                print()
                invalid += 1
            elif response == test_label:
                correct += 1
            else:
                incorrect += 1
            total += 1

            # descriptive stats
            tp = np.array([1 if response == 1 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
            fp = np.array([1 if response == 1 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
            tn = np.array([1 if response == 0 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
            fn = np.array([1 if response == 0 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
            recall = tp/(tp + fn)
            precision = tp/(tp + fp)
            f1 = 2 * (precision * recall) / (precision + recall)

            if verbose:
                try: # handle divide by zero
                    print('Details: ' + str({
                        'Accuracy': round(correct/(incorrect + correct), 3), 
                        'Correct': correct, 
                        'Incorrect': incorrect, 
                        'Invalid': invalid, 
                        'API Timeout': api_timeout, 
                        'Total': total, 
                        'TP': tp, 
                        'FP': fp, 
                        'TN': tn, 
                        'FN': fn, 
                        'F1': round(f1, 3), 
                        'Recall': round(recall, 3), 
                        'Precision': round(precision, 3)
                    }))
                except:
                    print('Accuracy:', 0, 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'API Timeout:', api_timeout, 'Total:', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', 0, 'Recall:', 0, 'Precision:', 0)

    return {'hypothesis': hypothesis, 'responses': responses, 'gts': gts, 'correct': correct, 'incorrect': incorrect, 'invalid': invalid, 'api_timeout': api_timeout, 'total': total, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': round(correct/(incorrect + correct), 3)}

