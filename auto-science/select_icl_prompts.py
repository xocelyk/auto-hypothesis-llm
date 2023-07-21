import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

'''
Given N samples and constrained to n < N samples for in-context learning, select n samples from N samples
Fit a logistic regression model on the N samples, and choose the n that maximize entropy (highest uncertainty)
'''

def entropy_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T
    # print all the rows in train_data that have na

    train_data.dropna(inplace=True)
    train_data = train_data.sample(frac=1)  # shuffle
    label = 'Label'
    # run logistic regression
    X = train_data.drop(label, axis=1)
    y = train_data[label]
    # print num rows in X that have na
    print('num rows in X that have na: ', X.isna().sum().sum())

    clf = LogisticRegression(random_state=0).fit(X, y)
    # get entropy
    train_data['pred_prob'] = clf.predict_proba(X)[:, 1]
    train_data['pred_prob'] = train_data['pred_prob'].apply(lambda x: min(x, 1 - x))
    train_data.sort_values(by='pred_prob', inplace=True, ascending=False)
    top_indices = train_data.index[:sample_size]
    res = {k: v for k, v in train_data_dict.items() if k in top_indices}
    return res

def active_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T

    train_data.dropna(inplace=True)
    train_data = train_data.sample(frac=1)  # shuffle
    label = 'Label'

    seed = 4
    # get 4 random samples
    seed_indices = train_data.sample(n=seed).index
    valid_data = train_data.drop(seed_indices)

    def get_max_entropy(df):
        df['entropy'] = df['pred_prob'].apply(lambda x: min(x, 1 - x))
        df.sort_values(by='entropy', inplace=True, ascending=False)
        return df.index[0]

    while valid_data.shape[0] < sample_size:
    # run logistic regression
        X = valid_data.drop(label, axis=1)
        y = valid_data[label]
        clf = LogisticRegression(random_state=0).fit(X, y)
        # get entropy
        valid_data['pred_prob'] = clf.predict_proba(X)[:, 1]
        print('num training samples: ', valid_data.shape[0], 'accuracy: ', clf.score(X, y))
        max_entropy_index = get_max_entropy(valid_data)
        # swap from valid_data to train_data
        train_data = train_data.append(valid_data.loc[max_entropy_index])
        valid_data = valid_data.drop(max_entropy_index)
    train_data_indices = train_data.index
    return {k: v for k, v in train_data_dict.items() if k in train_data_indices}

def cluster_learn_sample(train_data_dict, sample_size):
    assert sample_size < len(train_data_dict), 'sample_size must be less than the number of samples'
    train_data = pd.DataFrame(train_data_dict).T

    train_data.dropna(inplace=True)
    train_data = train_data.sample(frac=1)  # shuffle

    # Note: Here we are assuming that the 'Label' column is not considered
    # in the clustering process. If it should be, remove the next line.
    train_data_no_label = train_data.drop('Label', axis=1)

    # scale the features
    scaler = StandardScaler()
    train_data_scaled = pd.DataFrame(scaler.fit_transform(train_data_no_label), columns=train_data_no_label.columns, index=train_data_no_label.index)

    # run K-means clustering
    kmeans = KMeans(n_clusters=sample_size, random_state=0).fit(train_data_scaled)

    # get cluster centers
    cluster_centers = kmeans.cluster_centers_
    
    # find the closest data point from each cluster center
    # we will store indices of closest points here
    closest_points_indices = []
    for center in cluster_centers:
        distances = ((train_data_scaled - center)**2).sum(axis=1)
        closest_points_indices.append(distances.idxmin())

    # Filter the dictionary to return only those keys that are in closest_points_indices
    res = {k: v for k, v in train_data_dict.items() if k in closest_points_indices}

    return res





# train_data_dict = pickle.load(open('data/experiment1/train_data.pkl', 'rb'))
# res = select_icl_prompts(train_data_dict, 10)
# print(res)
