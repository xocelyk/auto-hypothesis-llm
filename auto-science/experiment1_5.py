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
1 shot hypothesis, 0 icl shots
'''

HYPOTHESIS = '''If median income is greater than $68,000, 
    then the median house value for houses in this block is greater than $200,000.
Else if housing median age is less than or equal to 15.0, 
    then the median house value for houses in this block is greater than $200,000.
Else if total rooms is greater than 4274.0, 
    then the median house value for houses in this block is greater than $200,000.
Else,
    the median house value for houses in this block is less than or equal to $200,000. 

This decision tree hypothesis suggests that the median income is the most important feature in determining the median house value for houses in a block. Higher median income is associated with higher median house values. If median income is not high enough, then the housing median age and total rooms become important predictors.'''

def save_state(state_dict):
    with open('data/experiment1_5_results.pkl', 'wb') as f:
        pickle.dump(state_dict, f)

def load_state():
    with open('data/experiment1_5_results.pkl', 'rb') as f:
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

    num_icl_shots = 1
    # hypothesis_num_shots = 64
    hypothesis = HYPOTHESIS

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
    # hypothesis_num_shots = 1
    # hypotheses = get_hypothesis(train_data, gen_hypothesis_temperature, hypothesis_num_shots, 1)
    # print(hypotheses[0])
    reset_state()
    with Manager() as manager:
        # Create a proxy for the shared state dictionary
        state_dict_proxy = manager.dict(load_state())
        
        # Start the runner with the main function and the state dictionary proxy
        runner(main, state_dict_proxy)

