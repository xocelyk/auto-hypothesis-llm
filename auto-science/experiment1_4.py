from config import load_config
from data_loader import split_data
import numpy as np
from few_shot import few_shot
from hypothesize import get_hypothesis
from test import test_hypothesis, test_hypothesis_one_example
import pickle
import datetime
from multiprocessing import Process, Manager

'''
16 shot hypothesis, 0 icl shot
'''

HYPOTHESIS = '''- If the housing median age is less than or equal to 15, then the median house value for houses in this block is greater than $200,000.
- If the housing median age is greater than 15 and the median income is greater than $40,000, then the median house value for houses in this block is greater than $200,000.
- If the housing median age is greater than 15 and the median income is less than or equal to $40,000, then:
  - If the total rooms is greater than 2000, then the median house value for houses in this block is greater than $200,000.
  - If the total rooms is less than or equal to 2000, then the median house value for houses in this block is less than or equal to $200,000.

This decision tree hypothesis suggests that the housing median age and the median income are the most important factors in predicting whether the median house value in a block is greater than $200,000. If the median age is less than or equal to 15, then the median house value is likely to be greater than $200,000. If the median age is greater than 15, then the median income is used as a criterion. If the median income is greater than $40,000, then the median house value is likely to be greater than $200,000. If the median income is less than or equal to $40,000, then the total rooms is a secondary criterion. If the total rooms is greater than 2000, then the median house value is likely to be greater than $200,000. If the total rooms is less than or equal to 2000, then the median house value is likely to be less than or equal to $200,000.'''


def save_state(state_dict):
    with open('data/experiment1_4_results.pkl', 'wb') as f:
        pickle.dump(state_dict, f)

def load_state():
    with open('data/experiment1_4_results.pkl', 'rb') as f:
        return pickle.load(f)

def reset_state():
    test_validation_data_keys = pickle.load(open('data/experiment1/test_validation_data_keys.pkl', 'rb'))
    state_dict = {'correct': 0, 'incorrect': 0, 'total': 0, 'invalid': 0, 'test_keys_remaining': test_validation_data_keys, 'test_keys_used': []}
    save_state(state_dict)

def runner(main_func, state_dict_proxy):
    while len(state_dict_proxy['test_keys_remaining']) > 0:
        proc = Process(target=main_func)
        proc.start()
        proc.join(timeout=10)
        if proc.is_alive():
            print('TIMEOUT')
            proc.terminate()
            proc.join()
            continue

def main():
    # gen_hypothesis_temperature = 0.7
    # num_hypotheses = 1
    # test_size = 500 # for test, change later
    config = load_config()
    data_mode, data_dict, data_frame, prompts = config['data_mode'], config['data_dict'], config['data_frame'], config['prompts']

    train_data_filename = 'data/experiment1/train_data.pkl'
    test_icl_data_filename = 'data/experiment1/test_icl_data.pkl'
    test_validation_data_filename = 'data/experiment1/test_validation_data.pkl'

    # save to pickle
    train_data = pickle.load(open(train_data_filename, 'rb'))
    test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    state_dict_proxy = load_state()
    test_keys_remaining = state_dict_proxy['test_keys_remaining']
    test_key = test_keys_remaining[0]

    num_icl_shots = 0
    # hypothesis_num_shots = 64
    hypothesis = '- If the median income for households in a block is greater than $46,000,\n    - And the housing median age is less than or equal to 33,\n        - And the total number of rooms in the block is greater than 1850,\n            - Then the median house value for houses in this block is greater than $200,000.\n    - And the housing median age is greater than 33,\n        - Then the median house value for houses in this block is less than or equal to $200,000.\n- If the median income for households in a block is less than or equal to $46,000,\n    - Then the median house value for houses in this block is less than or equal to $200,000.\n\nThis decision tree hypothesis suggests that the median income would be the most important factor in predicting the median house value. If the median income is low, the median house value is also low. However, if the median income is high, other factors such as housing median age and total number of rooms become important in predicting the median house value.'

    result = test_hypothesis_one_example(hypothesis, num_shots=num_icl_shots, test_icl_data=test_icl_data, test_validation_data=test_validation_data[test_key])

    if result == 1:
        state_dict_proxy['correct'] += 1
    elif result == 0:
        state_dict_proxy['incorrect'] += 1
    elif result == -1:
        state_dict_proxy['invalid'] += 1

    state_dict_proxy['total'] += 1
    state_dict_proxy['test_keys_remaining'] = state_dict_proxy['test_keys_remaining'][1:]
    state_dict_proxy['test_keys_used'] = state_dict_proxy['test_keys_used'] + [test_key]
    print('correct:', state_dict_proxy['correct'], 'incorrect:', state_dict_proxy['incorrect'], 'invalid:', state_dict_proxy['invalid'], 'total:', state_dict_proxy['total'], 'test keys remaining:', len(state_dict_proxy['test_keys_remaining']), 'test keys used:', len(state_dict_proxy['test_keys_used']))
    save_state(state_dict_proxy)



if __name__ == '__main__':
    # train_data = pickle.load(open('data/experiment1/train_data.pkl', 'rb'))
    # gen_hypothesis_temperature = 0.7
    # hypothesis_num_shots = 16
    # hypotheses = get_hypothesis(train_data, gen_hypothesis_temperature, hypothesis_num_shots, 1)
    # print(hypotheses[0])
    reset_state()
    with Manager() as manager:
        # Create a proxy for the shared state dictionary
        state_dict_proxy = manager.dict(load_state())
        
        # Start the runner with the main function and the state dictionary proxy
        runner(main, state_dict_proxy)

