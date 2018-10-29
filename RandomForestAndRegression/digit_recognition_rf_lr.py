import numpy as np
import matplotlib.pyplot as plt

from struct import unpack
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier


def data_read(image, label):
    images = open(image, 'rb')
    labels = open(label, 'rb')
    images.read(4)
    n_images = images.read(4)
    n_images = unpack('>I', n_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    columns = images.read(4)
    columns = unpack('>I', columns)[0]
    labels.read(4)
    N = labels.read(4)
    N = unpack('>I', N)[0]
    img = np.zeros((N, rows * columns), dtype=np.uint8)
    label = np.zeros(N, dtype=np.uint8)
    for i in range(N):
        for j in range(rows * columns):
            pixel_val = images.read(1)
            img[i][j] = unpack('>B', pixel_val)[0]
        label_val = labels.read(1)
        label[i] = unpack('>B', label_val)[0]

    images.close()
    labels.close()
    return (img, label)

def logistic_regression(train_imgs,train_labels,test_imgs,test_labels):
    lr = LogisticRegression(solver='lbfgs')
    lr.fit(train_imgs, train_labels)

    test_result = lr.predict(test_imgs)
    result_array = np.zeros((len(test_result), 10))
    for r in range(len(test_result)):
        x = test_result[r]
        result_array[r][x] = 1

    print("test result for Logistic Regression:",test_result)
    np.savetxt('lr.csv', (result_array), delimiter=',', fmt='%d')

    score = lr.score(test_imgs, test_labels)
    print("Test score for Logistic Regression:",score)


def random_forest(train_imgs,train_labels,test_imgs,test_labels):
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, max_features='auto', random_state=50)
    rf.fit(train_imgs, train_labels)

    test_result_rf = rf.predict(test_imgs)
    result_array_rf = np.zeros((len(test_result_rf), 10))
    for t in range(len(test_result_rf)):
        c = test_result_rf[t]
        result_array_rf[t][c] = 1
    print("len of result_rf=", len(test_result_rf))

    print("test result rf:", test_result_rf)
    np.savetxt('rf.csv', (result_array_rf), delimiter=",", fmt='%d')

    score_rf = rf.score(test_imgs, test_labels)
    print("Test Score for Random Forest",score_rf)

if __name__ == "__main__":
    train_imgs, train_labels = data_read('Data/train-images-idx3-ubyte', 'Data/train-labels-idx1-ubyte')
    print("train images shape: ", train_imgs.shape)
    print("train labels shape:", train_labels.shape)

    test_imgs, test_labels = data_read('Data/t10k-images-idx3-ubyte'
                                       , 'Data/t10k-labels-idx1-ubyte')
    print("test images shape: ", test_imgs.shape)
    print("test labels shape:", test_labels.shape)

    logistic_regression(train_imgs, train_labels, test_imgs, test_labels)

    random_forest(train_imgs, train_labels, test_imgs, test_labels)

    np.savetxt('name.csv', ["srajend2"], fmt='%s')

