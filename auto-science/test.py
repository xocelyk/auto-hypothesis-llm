import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt, get_test_ts_label, parse_response
from dotenv import load_dotenv
import os
from load_prompts import load_prompts

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

prompts = load_prompts(filename='prompts/vitals.json')
NUM_SHOTS = 1

import multiprocessing

def get_response_with_timeout(prompt, temperature):
    return get_response(prompt, temperature)


def test_hypothesis(hypothesis, test_icl_data=None, test_validation_data=None, verbose=True):
    # set up for stat collection
    first = True
    correct = 0
    incorrect = 0
    invalid = 0
    total = 0

    # prompt set up
    system_content = prompts['SYSTEM_CONTENT_1']
    user_content_1 = prompts['USER_CONTENT_1']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    hypothesis_prompt = 'Hypothesis: ' + hypothesis

    # experiment
    test_keys = list(test_validation_data.keys())
    np.random.shuffle(test_keys)
    responses = []
    gts = []
    for key in test_keys:
        # TODO: messy and repetitive
        test_sample = test_validation_data[key]
        test_ts, test_label = get_test_ts_label(test_sample)
        messages = [{"role": "system", "content": system_content}, {"role": "user", "content": user_content_1}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}, {"role": "user", "content": hypothesis_prompt}]
        prompt = create_prompt(num_shots=NUM_SHOTS, train_data=test_icl_data, test_data=test_sample, train_mode=True, test_mode=True, messages=messages)
        last_message = prompt.pop()
        prompt.append({'role': 'user', 'content': 'As a reminder, this is the hypothesis: ' + hypothesis})
        prompt.append(last_message)
        if first and verbose:
            for el in prompt:
                print(el['content'])
                print()
        # Initialize a Pool with one process
        pool = multiprocessing.Pool(processes=1)

        # Call get_response_with_timeout() in that process, and set timeout as 5 seconds
        result = pool.apply_async(get_response_with_timeout, args=(prompt, 0.0))

        try:
            # get the result within 5 seconds
            response_text = result.get(timeout=5)[0]
        except multiprocessing.TimeoutError:
            print("get_response() function took longer than 5 seconds.")
            pool.terminate()  # kill the process
            continue  # go to the next loop iteration

        pool.close()  # we are not going to use this pool anymore
        pool.join()  # wait for the pool to close by joining
        if first and verbose:
            print(response_text)
        response = parse_response(response_text)
        responses.append(response)
        gts.append(test_label)
        if verbose:
            if first:
                first = False
            if response == -1:
                invalid += 1
            elif response == test_label:
                correct += 1
            else:
                incorrect += 1
            total += 1

            # descriptive stats
            # true positive
        tp = np.array([1 if response == 1 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
        # false positive
        fp = np.array([1 if response == 1 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
        # true negative
        tn = np.array([1 if response == 0 and test_label == 0 else 0 for response, test_label in zip(responses, gts)]).sum()
        # false negative
        fn = np.array([1 if response == 0 and test_label == 1 else 0 for response, test_label in zip(responses, gts)]).sum()
        recall = tp/(tp + fn)
        precision = tp/(tp + fp)
        f1 = 2 * (precision * recall) / (precision + recall)

        if verbose:
            try: # handle divide by zero
                print('Accuracy:', round(correct/(incorrect + correct), 3), 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'Total', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', round(f1, 3), 'Recall:', round(recall, 3), 'Precision:', round(precision, 3))
            except:
                print('Accuracy:', 0, 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'Total', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', 0, 'Recall:', 0, 'Precision:', 0)
    return {'responses': responses, 'gts': gts, 'correct': correct, 'incorrect': incorrect, 'invalid': invalid, 'total': total, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': round(correct/(incorrect + correct), 3)}

