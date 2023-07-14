from ..config import load_config
from ..test import test_hypothesis_one_example
import pickle
from multiprocessing import Process, Manager

def save_state(state_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f)

def load_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def reset_state():
    test_validation_data_keys = pickle.load(open('../data/experiment1/test_validation_data_keys.pkl', 'rb'))
    state_dict = {'correct': 0, 'incorrect': 0, 'total': 0, 'invalid': 0, 'test_keys_remaining': test_validation_data_keys, 'test_keys_used': []}
    save_state(state_dict)

def runner(main_func, state_dict_proxy, *args, **kwargs):
    while len(state_dict_proxy['test_keys_remaining']) > 0:
        proc = Process(target=main_func, args=args, kwargs=kwargs)
        proc.start()
        proc.join(timeout=10)
        if proc.is_alive():
            print('TIMEOUT')
            proc.terminate()
            proc.join()
            continue

def main(num_icl_shots, hypothesis, filename):
    config = load_config()
    data_mode, data_dict, data_frame, prompts = config['data_mode'], config['data_dict'], config['data_frame'], config['prompts']

    train_data_filename = '../data/experiment1/train_data.pkl'
    test_icl_data_filename = '../data/experiment1/test_icl_data.pkl'
    test_validation_data_filename = '../data/experiment1/test_validation_data.pkl'

    # save to pickle
    train_data = pickle.load(open(train_data_filename, 'rb'))
    test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    state_dict_proxy = load_state(filename)
    test_keys_remaining = state_dict_proxy['test_keys_remaining']
    test_key = test_keys_remaining[0]

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
    state_dict_proxy['hypothesis'] = hypothesis
    state_dict_proxy['num_icl_shots'] = num_icl_shots
    print('correct:', state_dict_proxy['correct'], 'incorrect:', state_dict_proxy['incorrect'], 'invalid:', state_dict_proxy['invalid'], 'total:', state_dict_proxy['total'], 'test keys remaining:', len(state_dict_proxy['test_keys_remaining']), 'test keys used:', len(state_dict_proxy['test_keys_used']))
    save_state(state_dict_proxy, filename)

def test(num_icl_shots, hypothesis, filename):
    reset_state()
    with Manager() as manager:
        # Create a proxy for the shared state dictionary
        state_dict_proxy = manager.dict(load_state(filename))
        
        # Start the runner with the main function and the state dictionary proxy
        runner(main, state_dict_proxy, num_icl_shots=num_icl_shots, hypothesis=hypothesis, filename=filename)

