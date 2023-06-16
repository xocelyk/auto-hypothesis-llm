import pickle
import numpy as np
import pandas as pd

def load_data():
    # data_dict = pickle.load(open('../data/weather.pkl', 'rb'))
    data_dict = pickle.load(open('../data/vitals.pkl', 'rb'))


    data_dict_keys = list(data_dict.keys())
    np.random.shuffle(data_dict_keys)

    train_keys = data_dict_keys[:int(len(data_dict_keys)*0.3)]
    test_icl_keys = data_dict_keys[int(len(data_dict_keys)*0.3):]
    test_validation_keys = data_dict_keys[int(len(data_dict_keys)*0.3):int(len(data_dict_keys)*0.4)]
    train_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in train_keys}
    test_icl_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in test_icl_keys}
    test_validation_data = {data_dict_key: data_dict[data_dict_key] for data_dict_key in test_validation_keys}
    return train_data, test_icl_data, test_validation_data

if __name__ == '__main__':
    DATA_TYPE = 'vitals'
    if DATA_TYPE == 'weather':
        data_dir = '../data/'
        filename = 'city_temperature.csv'
        df = pd.read_csv(data_dir + filename)
        df['Region'].value_counts()/len(df)
        yes_label = 'North America'
        df['Label'] = df.apply(lambda x: 1 if x['Region'] == yes_label else 0, axis=1)
        df.replace(-99.0, np.nan, inplace=True)
        df.dropna(subset=['AvgTemperature'], inplace=True)
        df['State'].fillna('NA', inplace=True)
        df = df.groupby(['Year', 'Region', 'Country', 'State', 'City'])['AvgTemperature'].apply(list)
        df = df.reset_index()
        df['Label'] = df.apply(lambda x: 1 if x['Region'] == yes_label else 0, axis=1)
        df['len_ts'] = df['AvgTemperature'].apply(lambda x: len(x))

        def day_ts_to_month_ts(ts):
            months = [0, 31, 31+28, 31+28+31, 31+28+31+30, 31+28+31+30+31, 31+28+31+30+31+30, 31+28+31+30+31+30+31, 31+28+31+30+31+30+31+31, 31+28+31+30+31+30+31+31+30, 31+28+31+30+31+30+31+31+30+31, 31+28+31+30+31+30+31+31+30+31+30, 31+28+31+30+31+30+31+31+30+31+30+31]
            if len(ts) == 366: # leap year
                for i in range(2, 13):
                    months[i] += 1
            elif len(ts) != 365:
                return None
            month_ts = []
            for i in range(1, 13):
                month_ts.append(int(round(np.mean(ts[months[i-1]:months[i]]), 0)))
            return month_ts

        df['month_ts'] = df['AvgTemperature'].apply(day_ts_to_month_ts)
        df['month_ts'] = df['AvgTemperature'].apply(day_ts_to_month_ts)
        df.dropna(inplace=True)
        df[['temp1', 'temp2', 'temp3', 'temp4', 'temp5', 'temp6', 'temp7', 'temp8', 'temp9', 'temp10', 'temp11', 'temp12']] = pd.DataFrame(df['month_ts'].tolist(), index=df.index)
        data_dict = df[['Year', 'Region', 'Country', 'State', 'City', 'month_ts', 'Label']].rename(columns={'month_ts': 'MonthlyAvgTemperature'}).to_dict(orient='index')

        with open('../data/weather.pkl', 'wb') as handle:
            pickle.dump(data_dict, handle)

    if DATA_TYPE == 'vitals':
        # TODO: manually specify units?
        data_dict = pickle.load(open('../data/vitals_raw.pkl', 'rb'))
        d_items = pd.read_csv('/Users/kylecox/Documents/ws/mimic-llm/prompting/data/d_items.csv')
        d_items_dict = {d_items['itemid'][i]: d_items['label'][i] for i in range(len(d_items))}
        new_data_dict = {}
        for key in data_dict:
            new_data_dict[key] = {}
            for label, val in data_dict[key].items():
                if type(val) == list:
                    new_label = d_items_dict[int(label)]
                    new_list = [int(round(el)) for el in val]
                    new_list = [el if el != 0 else 'NA' for el in new_list]
                    new_data_dict[key][new_label] = new_list
                elif label == 'label':
                    new_data_dict[key]['Label'] = val
                else:
                    new_data_dict[key][label] = val
        with open('../data/vitals.pkl', 'wb') as handle:
            pickle.dump(new_data_dict, handle)
