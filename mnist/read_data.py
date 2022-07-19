import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

# read data from file
def from_file(dataset="mnist", type="original", plot=False):
    print("dataset: {}, type: {}".format(dataset, type))
    (X_train_org, y_train_org), (X_test_org, y_test_org) = mnist.load_data()
    # get no of training samples
    n_train_org = len(y_train_org)
    # get no of testing samples
    n_test_org = len(y_test_org)
    # get no of classes
    n_class = len(np.unique(y_train_org))
    # get image size
    img_rows, img_cols = X_train_org.shape[1:]
    # get no of features
    n_feature = img_rows * img_cols
    # get image shape
    img_shape = (img_rows, img_cols, 1)
    print("original data")
    print("n_train_org: {}, n_feature: {}, img_shape: {}, n_class: {}".format(n_train_org, n_feature, img_shape, n_class))
    for label in range(n_class):
        n_label = len(np.where(y_train_org == label)[0])
        print("label: {}, n_label: {}".format(label, n_label))
    print("n_test_org: {}".format(n_test_org))
    n_labels_test = []
    for label in range(n_class):
        n_label = len(np.where(y_test_org == label)[0])
        print("label: {}, n_label: {}".format(label, n_label))
        n_labels_test.append(n_label)
    if type == "original":
        # keep original data
        X_train = X_train_org
        y_train = y_train_org
    if type == "subset":
        # keep balanced data but sample a subset (200 data points) of each label
        n_label_idx = 200
        indices_all_labels = np.array([], dtype=int)
        for label in range(n_class):
            indices_label = np.where(y_train_org == label)[0]
            np.random.shuffle(indices_label)
            indices_all_labels = np.concatenate((indices_all_labels, indices_label[:n_label_idx]))
        # need to shuffle otherwise the samples with labels are in sequential 0, 1, 2,...
        np.random.shuffle(indices_all_labels)
        X_train = X_train_org[indices_all_labels]
        y_train = y_train_org[indices_all_labels]
    print("modified data")
    n_train = len(y_train)
    print("n_train: {}, n_feature: {}, img_shape: {}, n_class: {}".format(n_train, n_feature, img_shape, n_class))
    n_labels_train = []
    for label in range(n_class):
        n_label = len(np.where(y_train == label)[0])
        print("label: {}, n_label: {}".format(label, n_label))
        n_labels_train.append(n_label)
    if plot == True:
        labels = range(n_class)
        plt.bar(labels, n_labels_train)
        plt.xticks(range(n_class))
        plt.xlabel("label")
        plt.ylim(0, 7000)
        plt.ylabel("# of samples")
        plt.savefig("./{}_{}_train.pdf".format(dataset, type), bbox_inches="tight")
        plt.close()
        plt.bar(labels, n_labels_test)
        plt.xticks(range(n_class))
        plt.xlabel("label")
        plt.ylim(0, 7000)
        plt.ylabel("# of samples")
        plt.savefig("./{}_{}_test.pdf".format(dataset, type), bbox_inches="tight")
        plt.close()

    return X_train, y_train, X_test_org, y_test_org, n_train, n_test_org, n_feature, n_class, img_shape


# X_train, y_train, X_test, y_test, n_train, n_test, n_feature, n_class, img_shape = from_file("mnist", "original", plot=True)

