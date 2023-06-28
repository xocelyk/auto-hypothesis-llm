from data_loader import load_data
from hypothesize import get_hypothesis
from test import test_hypothesis
import numpy as np

if __name__ == '__main__':

    train_data, test_icl_data, test_validation_data = load_data()
    hypothesis_temperature = 0.7
    sample_size = 50
    test_size = 200
    test_validation_data_keys = list(test_validation_data.keys())
    np.random.shuffle(test_validation_data_keys)
    test_validation_data_keys = test_validation_data_keys[:test_size]
    test_validation_data = {key: test_validation_data[key] for key in test_validation_data_keys}

    # get hypothesis
    hypotheses = get_hypothesis(train_data, hypothesis_temperature, sample_size, 1)
    # write hypothesis to text file
    with open('hypothesis.txt', 'w') as f:
        for i, el in enumerate(hypotheses):
            f.write('Hypothesis ' + str(i + 1) + ': ')
            f.write(el)
            f.write('\n\n')

    # hypotheses = ['']

    for hypothesis in hypotheses:
        print()
        print(hypothesis)
        print()
        # get user input on whether or not to continue
        # if user says no, then we're done
        input1 = input("Continue? (y/n) ")
        if input1 == 'n':
            continue
        print()

        test_results = test_hypothesis(hypothesis, test_icl_data, test_validation_data)
        print({k: v for k, v in test_results.items() if k in ['correct', 'incorrect', 'total', 'invalid', 'accuracy', 'f1']})
