from data_loader import load_data
from hypothesize import get_hypothesis
from test import test_hypothesis
import numpy as np

if __name__ == '__main__':

    train_data, test_icl_data, test_validation_data = load_data()
    hypothesis_temperature = 0.7
    sample_size = 15
    test_size = 100
    test_validation_data_keys = list(test_validation_data.keys())
    np.random.shuffle(test_validation_data_keys)
    test_validation_data_keys = test_validation_data_keys[:test_size]
    test_validation_data = {key: test_validation_data[key] for key in test_validation_data_keys}

    # get hypothesis
    hypotheses = get_hypothesis(train_data, hypothesis_temperature, sample_size, 5)
    print(len(hypotheses))
    # hypothesis = "Based on the given patient data, I have provided the following hypotheses:\n\
    # - (A) If the average heart rate of the patient during the first 24 hours of their ICU stay is high and the average respiratory rate is low, then their ICU stay was longer than 3 days.\n\
    # - (B) If the average heart rate of the patient during the first 24 hours of their ICU stay is low and the average respiratory rate is high, then their ICU stay was shorter than 3 days.\n\
    # - (B) If the patient's O2 saturation pulse oximetry readings are consistently high throughout the first 24 hours of their ICU stay, then their ICU stay was shorter than 3 days.\n\
    # - (A) If the patient's non-invasive blood pressure mean and systolic readings are consistently high throughout the first 24 hours of their ICU stay, then their ICU stay was longer than 3 days."
    # hypothesis = "Based on the data provided, I can hypothesize that cities in North America tend to have a wider range of temperatures throughout the year compared to cities outside of North America. This is because North America experiences four distinct seasons, while other regions may have less variation in temperature throughout the year. Additionally, cities in North America tend to have colder winters and hotter summers compared to cities outside of North America. However, it is important to note that this hypothesis is based on a limited set of data and may not be applicable to all cities in North America or outside of North America. As more data is collected, this hypothesis can be refined and improved."
    # hypothesis = 'North American cities are very cold. Only those cities that are very cold throughout the year are in North America. Further, only cities with low variance in temperature are in North America. Finally, only cities with low variance in temperature are in North America.'

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
