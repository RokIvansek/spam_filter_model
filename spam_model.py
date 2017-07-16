import numpy as np
import urllib.request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def parse_data(save_to_cwd=False):
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
    if save_to_cwd:
        np.save("X_raw", np.array(X))
        np.save("Y_raw", np.array(Y))
    return X, Y

def load_from_npy_file(path_to_X_and_Y):
    X = np.load(path_to_X_and_Y + "/X_raw.npy")
    Y = np.load(path_to_X_and_Y + "/Y_raw.npy")
    return X, Y

def split_train_test(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    return (X_train, X_test, y_train, y_test)

def extract_features(X_train):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    # print(X_train_counts.shape)
    return X_train_counts, count_vect

# This makes the counts too close to zero and pca doesn't work then
# TODO: You should find a smarter way to incorporate length of string into features
def devide_by_log_length_of_string(X, X_features):
    n = len(X)
    # print(n)
    # X_features_fixed = np.copy(X_features)
    # print(X_features.shape)
    for i in range(n):
        X_features[i,:] = X_features[i,:]/np.log(len(X[i]))
    return X_features

def standard_scaler(X_train):
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train.toarray())
    return X_train_scaled, ss

def pca(X_train, n_comp=100):
    pca_processor = PCA(n_components=n_comp)
    X_train_pca = pca_processor.fit_transform(X_train)
    # print("Selected components explain " +
    #       str(int(100*np.round(np.sum(pca_processor.explained_variance_ratio_), decimals=2))) +
    #       " percent of the variance.")
    return X_train_pca, pca_processor

def pre_process(X, count_vect, ss, pca):
    if type(X) == str: # If input is a single string
        return pca.transform(ss.transform(count_vect.transform(np.array([X])).toarray()))
    else: # If we have a matrix
        return pca.transform(ss.transform(count_vect.transform(X).toarray()))

def train_model(X, y):
    rf = RandomForestClassifier(n_estimators=2000, n_jobs=-1)
    rf.fit(X, y)
    return rf

def predict_on_X_test(X_test, model):
    y_test_predict = model.predict(X_test)
    return y_test_predict

def predict_a_single_string(s, count_vect, ss, pca, model):
    s_preprocessed = pre_process(s, count_vect, ss, pca)
    return model.predict(s_preprocessed)[0]

if __name__ == '__main__':
    # X, Y = parse_data(save_to_cwd=True)
    X, Y = load_from_npy_file("/home/rok/Documents/projects/spam_filter_model")
    X_train, X_test, y_train, y_test = split_train_test(X, Y)
    X_train_features, count_vect = extract_features(X_train)
    # X_train_features_fixed = devide_by_log_length_of_string(X_train, X_train_features)
    X_train_scaled, ss = standard_scaler(X_train_features)
    X_train_pca, pca_processor = pca(X_train_scaled, 100)
    X_test_pca = pre_process(X_test, count_vect, ss, pca_processor)
    rf_model = train_model(X_train_pca, y_train)
    y_test_prediction = predict_on_X_test(X_test_pca, rf_model)
    print(accuracy_score(y_test, y_test_prediction))
    # Single string prediction
    s_genuine = "Pokemon master Rok is the best pokemon master"
    s_spam = "Call now and win a 2000 dollar prize or visit www.free2000dollars.com."
    predict_a_single_string(s_genuine, count_vect, ss, pca_processor, rf_model)
    predict_a_single_string(s_spam, count_vect, ss, pca_processor, rf_model)

