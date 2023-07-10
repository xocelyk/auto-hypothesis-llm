from config import load_config
from data_loader import split_data
import numpy as np
from few_shot import few_shot
from hypothesize import get_hypothesis
from test import test_hypothesis
import pickle


filename = 'experiment1_results.pkl'
experiment_results = pickle.load(open(filename, 'rb'))
# experiment_results = {}

def main():
    gen_hypothesis_temperature = 0.7
    num_hypotheses = 5
    test_size = 200 # for test, change later
    config = load_config()
    data_mode, data_dict, data_frame, prompts = config['data_mode'], config['data_dict'], config['data_frame'], config['prompts']
    # standardize train, test, valid datasets
    train_data, test_icl_data, test_validation_data = split_data(data_dict, 100, 100, test_size)

    # few-shot, no-hypothesis: 0, 1, 2, 4, 8, 16, 32
    num_icl_shots_lst = [0, 1, 4, 32]
    # for num_shots in num_icl_shots_lst:
    #     print('\n')
    #     print('Few shot, no hypothesis, num_shots:', num_shots)
    #     # TODO: make sure that icl data is being shuffled each iteration
    #     results = few_shot(train_data, test_validation_data, num_shots, verbose=True)
    #     print('Accuracy:', results['accuracy'])
    #     experiment_results[f'few-shot, no-hypothesis, {num_shots}'] = results
    #     with open('experiment1_results.pkl', 'wb') as f:
    #         pickle.dump(experiment_results, f)

    # hypothesis generation, decision tree, 5 each: 4, 16, 64
    # for each hypothesis:
        # few-shot, hypothesis: 0, 1, 2, 4, 8, 16, 32
    hypothesis_num_shots_lst = [4, 16, 64]
    for i in range(num_hypotheses):
        for hypothesis_num_shots in hypothesis_num_shots_lst:
            hypotheses = get_hypothesis(train_data, gen_hypothesis_temperature, hypothesis_num_shots, 1)
            for hypothesis in hypotheses:
                for num_icl_shots in num_icl_shots_lst[:2]:
                    print('\n')
                    print('Few shot, hypothesis, num shots hypothesis generation:', hypothesis_num_shots, 'num ICL shots:', num_icl_shots)
                    print()
                    print('Hypothesis:', hypothesis)
                    print()
                    results = test_hypothesis(hypothesis, num_shots=num_icl_shots, test_icl_data=test_icl_data, test_validation_data=test_validation_data, verbose=True)
                    print('Accuracy:', results['accuracy'])
                    experiment_results[f'hypothesis, {hypothesis_num_shots}, {num_icl_shots}, hypothesis_{i}'] = results
                    with open('experiment1_results.pkl', 'wb') as f:
                        pickle.dump(experiment_results, f)
    return experiment_results

if __name__ == '__main__':
    results = main()
    with open('experiment1_results.pkl', 'wb') as f:
       pickle.dump(results, f)
