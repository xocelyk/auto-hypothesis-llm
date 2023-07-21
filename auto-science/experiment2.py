from config import load_config
from test import test_hypothesis_one_example
import pickle
from multiprocessing import Process, Manager
from hypothesize import get_hypothesis
from select_icl_prompts import cluster_learn_sample
from few_shot import few_shot_one_example

def save_state(state_dict_proxy, filename):
    # Convert proxy dict to regular dict before pickling
    state_dict = dict(state_dict_proxy)
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f)


def load_state(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def reset_state(filename):
    test_validation_data_keys = pickle.load(open('data/experiment2/test_validation_data_keys.pkl', 'rb'))
    test_validation_data_keys = list(test_validation_data_keys)
    state_dict = {'correct': 0, 'incorrect': 0, 'total': 0, 'invalid': 0, 'test_keys_remaining': test_validation_data_keys, 'test_keys_used': []}
    save_state(state_dict, filename)


def runner(main_func, state_dict_proxy, *args, **kwargs):
    while len(state_dict_proxy['test_keys_remaining']) > 0:
        proc = Process(target=main_func, args=[state_dict_proxy, *args], kwargs=kwargs)  # Pass state_dict_proxy to the main function
        proc.start()
        proc.join(timeout=10)
        if proc.is_alive():
            print('TIMEOUT')
            proc.terminate()
            proc.join()
            continue


def few_shot(state_dict_proxy, num_icl_shots, filename):
    test_icl_data_filename = 'data/experiment2/test_icl_data.pkl'
    test_validation_data_filename = 'data/experiment2/test_validation_data.pkl'

    # save to pickle
    test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    test_keys_remaining = state_dict_proxy['test_keys_remaining']
    test_key = test_keys_remaining[0]

    result_dict = few_shot_one_example(num_shots=num_icl_shots, test_icl_data=test_icl_data, test_validation_data=test_validation_data[test_key])
    result = result_dict['correct']
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
    save_state(state_dict_proxy, filename)


def main(state_dict_proxy, num_icl_shots, hypothesis, filename):
    test_icl_data_filename = 'data/experiment2/test_icl_data.pkl'
    test_validation_data_filename = 'data/experiment2/test_validation_data.pkl'

    # save to pickle
    test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
    test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))

    test_keys_remaining = state_dict_proxy['test_keys_remaining']
    test_key = test_keys_remaining[0]

    result_dict = test_hypothesis_one_example(hypothesis, num_shots=num_icl_shots, test_icl_data=test_icl_data, test_validation_data=test_validation_data[test_key])
    result = result_dict['correct']
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
    save_state(state_dict_proxy, filename)

def test(num_icl_shots, hypothesis, filename):
    reset_state(filename)
    with Manager() as manager:
        # Create a proxy for the shared state dictionary
        state_dict_proxy = manager.dict(load_state(filename))
        
        # Start the runner with the main function and the state dictionary proxy
        runner(main, state_dict_proxy, num_icl_shots=num_icl_shots, hypothesis=hypothesis, filename=filename)

    results = load_state(filename)
    results['hypothesis'] = hypothesis
    results['num_icl_shots'] = num_icl_shots
    return results

def test_few_shot(num_icl_shots, filename):
    reset_state(filename)
    with Manager() as manager:
        state_dict_proxy = manager.dict(load_state(filename))
        runner(few_shot, state_dict_proxy, num_icl_shots, filename)
    
    results = load_state(filename)
    results['num_icl_shots'] = num_icl_shots
    return results

def write_hypotheses(gen_hypothesis_temperature, num_hypothesis_shots, num_hypotheses):
    sep = '<H>\n\n\n\n<H>'
    train_data = pickle.load(open('data/experiment2/train_data.pkl', 'rb'))
    train_data = cluster_learn_sample(train_data, num_hypothesis_shots)
    hypotheses = get_hypothesis(train_data, gen_hypothesis_temperature, num_hypothesis_shots, num_hypotheses)
    with open('data/experiment2/hypotheses/hypotheses_{}_shots.txt'.format(num_hypothesis_shots), 'w') as f:
        for hypothesis in hypotheses:
            print(hypothesis)
            print()
            f.write(str(hypothesis) + sep)

if __name__ == '__main__':
    import datetime

    # hypothesis testing
    num_hypothesis_shots_lst = [64, 16, 4, 1]
    for num_hypothesis_shots in num_hypothesis_shots_lst:
        num_icl_shots = 0
        with open('data/experiment2/hypotheses/hypotheses_{}_shots.txt'.format(num_hypothesis_shots), 'r') as f:
            hypotheses = f.read().split('<H>\n\n\n\n<H>')
            hypotheses = [h.strip() for h in hypotheses]
            for i, hypothesis in enumerate(hypotheses[:4]):
                hypothesis = '''- If the median income for households in a block is greater than $46,000
   - And the housing median age is less than or equal to 33,
        - And the total number of rooms in the block is greater than 1850,
            - Then the median house value for houses in this block is greater than $200,000.
    - And the housing median age is greater than 33,
       - Then the median house value for houses in this block is less than or equal to $200,000.
- If the median income for households in a block is less than or equal to $46,000,
   - Then the median house value for houses in this block is less than or equal to $200,000.

This decision tree hypothesis suggests that the median income would be the most important factor in predicting the median house value. If the median income is low, the median house value is also low. However, if the median income is high, other factors such as housing median age and total number of rooms become important in predicting the median house value.'''
                state_dict_filename = 'data/experiment2/state_dict_{}_shots_{}.pkl'.format(num_hypothesis_shots, i)
                print(hypothesis)
                results = test(num_icl_shots=num_icl_shots, hypothesis=hypothesis, filename=state_dict_filename)
                print(results)
                timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                # save_filename = 'data/experiment2/results_hypothesis_shots_{}_icl_shots_{}_hypothesis_num_{}_{}.pkl'.format(num_hypothesis_shots, num_icl_shots, i, timestamp)
                # save_state(results, save_filename)

    # few shot testing
    # num_icl_shots_lst = [1, 2, 4, 8, 16]
    # for num_shots in num_icl_shots_lst:
    #     state_dict_filename = 'data/experiment2/state_dict_few_shot_{}.pkl'.format(num_shots)
    #     results = test_few_shot(num_icl_shots=num_shots, filename=state_dict_filename)
    #     print(results)
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #     save_filename = 'data/experiment2/results_few_shot_{}_{}.pkl'.format(num_shots, timestamp)
    #     save_state(results, save_filename)
