from ..config import load_config
from ..data_loader import split_data
import numpy as np
from ..hypothesize import get_hypothesis
from ..test import test_hypothesis
import pickle
import datetime
from hypothesis_experiment import test

num_icl_shots_lst = [0, 1, 8]
num_hypothesis_shots_lst = [1, 8, 64]
num_hypotheses = 2

train_data_filename = 'data/experiment1/train_data.pkl'
test_icl_data_filename = 'data/experiment1/test_icl_data.pkl'
test_validation_data_filename = 'data/experiment1/test_validation_data.pkl'

train_data = pickle.load(open(train_data_filename, 'rb'))
test_icl_data = pickle.load(open(test_icl_data_filename, 'rb'))
test_validation_data = pickle.load(open(test_validation_data_filename, 'rb'))
gen_hypothesis_temperature = 0.7

for num_hypothesis_shots in num_hypothesis_shots_lst:
    for num_icl_shots in num_icl_shots_lst:
        for i in range(num_hypotheses):
            filename = f'results/hypothesis_{i}_hypothesis_shots_{num_hypothesis_shots}_icl_shots_{num_icl_shots}.pkl'
            hypothesis = get_hypothesis(train_data, gen_hypothesis_temperature, num_hypothesis_shots, 1)[0]
            test(num_icl_shots=num_icl_shots, hypothesis=hypothesis, filename=filename)

