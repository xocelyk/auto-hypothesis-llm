from data_loader import split_data
from hypothesize import get_hypothesis
from test import test_hypothesis
import numpy as np
from config import load_config


if __name__ == '__main__':
    config = load_config()
    data_mode, data_dict, data_frame, prompts = config['data_mode'], config['data_dict'], config['data_frame'], config['prompts']
    hypothesis_temperature = 0.7
    sample_size = 64
    train_size = 100
    test_icl_size = 100
    test_size = 200
    train_data, test_icl_data, test_validation_data = split_data(data_dict, train_size, test_icl_size, test_size)
    test_validation_data_keys = list(test_validation_data.keys())
    np.random.shuffle(test_validation_data_keys)
    test_validation_data_keys = test_validation_data_keys[:test_size]
    test_validation_data = {key: test_validation_data[key] for key in test_validation_data_keys}

    # get hypothesis
    hypotheses = get_hypothesis(train_data, hypothesis_temperature, sample_size, 1)
    # write hypothesis to text file
    with open(f'data/generated_prompts/hypothesis_{data_mode}.txt', 'w') as f:
        for i, el in enumerate(hypotheses):
            f.write('Hypothesis ' + str(i + 1) + ': ')
            f.write(el)
            f.write('\n\n')

    for hypothesis in hypotheses:
        # get user input on whether or not to continue
        # if user says no, then we're done
        print(hypothesis)
        input1 = input("Continue? (y/n) ")
        if input1 == 'n':
            continue
        print()
        test_results = test_hypothesis(hypothesis, num_shots=0, test_icl_data=test_icl_data, test_validation_data=test_validation_data, verbose=True)
        print({k: v for k, v in test_results.items() if k in ['correct', 'incorrect', 'total', 'invalid', 'accuracy', 'f1']})
