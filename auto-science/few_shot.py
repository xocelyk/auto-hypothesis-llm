import numpy as np
import pandas as pd
import numpy as np
import openai
from utils import get_response, create_prompt, parse_response
from dotenv import load_dotenv
import os
from load_prompts import load_prompts
from utils import load_data
import multiprocessing

def get_response_with_timeout(prompt, temperature):
    return get_response(prompt, temperature)

prompts = load_prompts(filename='prompts/titanic.json')

def cot_prompt(train_data, test_data, temperature=1, sample_size=16, num_hypotheses=1):
    system_content = prompts['SYSTEM_CONTENT_2']
    system_content = 'You are Charlie, a Titanic expert. You are studying passenger data from the Titanic disaster. Your job is to predict which passengers survived the Titanic shipwreck. Do you understand?'
    user_content_1 = prompts['USER_CONTENT_1']
    assistant_content_1 = prompts['ASSISTANT_CONTENT_1']
    user_content_2 = prompts['USER_CONTENT_2']
    user_content_3 = prompts['USER_CONTENT_3']
    messages = [{"role": "system", "content": system_content}, {"role": "assistant", "content": assistant_content_1}, {"role": "user", "content": user_content_2}]
    prompt = create_prompt(sample_size, train_data=train_data, test_data=test_data, messages=messages, train_mode=True, test_mode=True)
    prompt.append({"role": "user", "content": user_content_3})
    
    return prompt

def expreriment(sample_size=2, test_size=200):
    train_data, test_icl_data, test_validation_data = load_data()

    test_validation_keys = list(test_validation_data.keys())
    np.random.shuffle(test_validation_keys)
    test_validation_keys = test_validation_keys[:test_size]
    test_validation_data = {key: test_validation_data[key] for key in test_validation_keys}

    responses = []
    gts = []
    first = True
    correct = incorrect = invalid = total = 0
    for key in test_validation_data.keys():
        train_data_keys = list(train_data.keys())
        np.random.shuffle(train_data_keys)
        train_data_keys = train_data_keys[:sample_size]
        train_data = {key: train_data[key] for key in train_data_keys}
        test_label = test_validation_data[key]['Label']
        prompt = cot_prompt(train_data, test_validation_data[key], temperature=0.7, sample_size=sample_size)
        if first:
            for el in prompt:
                print(el['content'])
            first = False
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
        response = parse_response(response_text)
        responses.append(response)
        gts.append(test_label)
        if response == -1:
            print(response_text)
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
        verbose = True
        if verbose:
            try: # handle divide by zero
                print('Accuracy:', round(correct/(incorrect + correct), 3), 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'Total', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', round(f1, 3), 'Recall:', round(recall, 3), 'Precision:', round(precision, 3))
            except:
                print('Accuracy:', 0, 'Correct:', correct, 'Incorrect:', incorrect, 'Invalid:', invalid, 'Total', total, 'TP:', tp, 'FP:', fp, 'TN:', tn, 'FN:', fn, 'F1:', 0, 'Recall:', 0, 'Precision:', 0)
    return {'correct': correct, 'incorrect': incorrect, 'invalid': invalid, 'total': total, 'f1': f1, 'recall': recall, 'precision': precision, 'accuracy': round(correct/(incorrect + correct), 3)}

if __name__ == '__main__':
    # load_dotenv()
    # openai.api_key = os.getenv('OPENAI_API_KEY')
    print(expreriment(sample_size=1, test_size=500))