import numpy as np
import urllib.request


def get_data():
    data_file = urllib.request.urlopen("https://s3-ap-southeast-1.amazonaws.com/mettl-prod/data-science/SMSCollection.txt")
    data = data_file.read()
    list_of_rows = data.decode("utf-8").split("\n")
    X = []
    Y = []
    i = 0
    for row in list_of_rows:
        if i == len(list_of_rows) - 1:
            break
        else:
            y, x = row.split("\t", 1)
            X.append(x)
            Y.append(y)
        i += 1
    return (np.array(X), np.array(Y))

def extract_features(X):
    # TODO: Find out what informative features you can extract from texts
    # TODO: This extractor should work on a single string or an array of strings
    pass

def train_model(X, y):
    # TODO: Train a model and return it
    pass

def predict_a_single_string(s, feature_extractor, model):
    pass
    # TODO: Use the feature extractor and the model to make a prediction about a string

if __name__ == '__main__':
    X, y = get_data()
    print(X[0])
    print(y[0])
    print(X[X.shape[0]-1])
    print(y[y.shape[0]-1])
