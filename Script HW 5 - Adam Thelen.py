import gzip
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scipy.stats as st

class MnistFileInfo:

    def __init__(self, mb=0, no_of_images=0, height=0, width=0, file_type="images"):
        self.magic_number = mb
        self.no_of_images = no_of_images
        self.height = height
        self.width = width
        self.file_type = file_type

    def get_mnist_file_info(self, buffer, file_type="images"):

        self.file_type = file_type
        convert_vector = np.asarray([256**3, 256**2, 256, 1])

        if self.file_type == "images":
            info = np.frombuffer(buffer.read(16), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)
            self.height = np.dot(info[8:12], convert_vector)
            self.width = np.dot(info[12:16], convert_vector)

        if self.file_type == "labels":
            info = np.frombuffer(buffer.read(8), dtype=np.uint8)  # the first 16 bytes contain the file's information
            self.magic_number = np.dot(info[0:4], convert_vector)
            self.no_of_images = np.dot(info[4:8], convert_vector)

    def get_bytes(self):
        '''Get the number of bytes containing data'''
        if self.file_type == "images":
            return self.no_of_images * self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images

    def get_dimension(self):
        '''Get the dimension of the data to be reshape in numpy'''
        if self.file_type == "images":
            return self.no_of_images, self.height * self.width

        if self.file_type == "labels":
            return self.no_of_images


def to_numpy_dataframe(bytestream, bytestream_info):
    '''Convert the byte stream to a numpy array based on the corresponding information matrix'''

    all_bytes = np.frombuffer(bytestream.read(bytestream_info.get_bytes()), dtype=np.uint8)
    data_frame = np.asarray(all_bytes).reshape(bytestream_info.get_dimension())

    return data_frame


def remove_zero_columns(numpy_array):
    zeros_index = []
    for i in range(numpy_array.shape[1]):
        if np.sum(numpy_array[:, i]) == 0:
            zeros_index.append(i)

    return np.delete(numpy_array, zeros_index, 1), zeros_index



def remove_middle_rows(images, labels):
    remove_index = []
    for i in range(images.shape[0]):
        if labels[i] not in [0, 9]:
            remove_index.append(i)

    images = np.delete(images, remove_index, 0)
    labels = np.delete(labels, remove_index, 0)

    return images, labels



def compute_accuracy(y, y_predict):

    indicator = np.where(y == y_predict, 1, 0)
    accuracy = np.sum(indicator) / y.shape[0]
    return accuracy




files = {
    "test_images": "./mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "./mnist/t10k-labels-idx1-ubyte.gz",
    "train_images": "./mnist/train-images-idx3-ubyte.gz",
    "train_labels": "./mnist/train-labels-idx1-ubyte.gz"
}

with gzip.open(files['train_images'], 'rb') as train_images, \
        gzip.open(files['train_labels'], 'rb') as train_labels, \
        gzip.open(files['test_images'], 'rb') as test_images, \
        gzip.open(files['test_labels'], 'rb') as test_labels:

    # Getting the information header of each file
    train_images_info = MnistFileInfo()
    train_images_info.get_mnist_file_info(train_images)

    train_labels_info = MnistFileInfo()
    train_labels_info.get_mnist_file_info(train_labels, file_type="labels")

    test_images_info = MnistFileInfo()
    test_images_info.get_mnist_file_info(test_images)

    test_labels_info = MnistFileInfo()
    test_labels_info.get_mnist_file_info(test_labels, file_type="labels")

    # convert the bytestream to numpy arrays
    train_images = to_numpy_dataframe(train_images, train_images_info)
    train_labels = to_numpy_dataframe(train_labels, train_labels_info)
    test_images = to_numpy_dataframe(test_images, test_images_info)
    test_labels = to_numpy_dataframe(test_labels, test_labels_info)


test_labels = test_labels.astype('float32').reshape((10000))
test_images = test_images.astype('float32')
train_labels = train_labels.astype('float32').reshape((60000))
train_images = train_images.astype('float32')




def remove_non_zero_nines(test_labels, train_labels):
    #test = np.concatenate((test_labels, test_images), axis=1)
    #train = np.concatenate((train_labels, train_images), axis=1)
    
    index_test = []
    for i in range(test_labels.shape[0]):
        if test_labels[i] not in [0,9]:
            index_test.append(i)
    
    index_train = []
    for i in range(train_labels.shape[0]):
        if train_labels[i] not in [0,9]:
            index_train.append(i)
    
    return index_test, index_train


def remove_zero_variance_columns(numpy_array):
    zero_variance_index = []
    for i in range(numpy_array.shape[1]):
        if np.var(numpy_array[:, i]) == 0:
            zero_variance_index.append(i)

    return np.delete(numpy_array, zero_variance_index, 1), zero_variance_index



##############################################################################


index_test, index_train = remove_non_zero_nines(test_labels, train_labels)

# Remove all numbers except 0 and 9
test_images_lg = np.delete(test_images, index_test, axis=0)
train_images_lg = np.delete(train_images, index_train, axis=0)
test_labels_lg = np.delete(test_labels, index_test, axis=0)
train_labels_lg = np.delete(train_labels, index_train, axis=0)

# replace the 9's with 1's
train_labels_lg = np.where(train_labels_lg == 9, 1, 0)
test_labels_lg = np.where(test_labels_lg == 9, 1, 0)

# Remove any columns that are only zeros
train_images_lg, removed_cols = remove_zero_columns(train_images_lg)
test_images_lg = np.delete(test_images_lg, removed_cols, 1)

# Remove columns with zero variance
train_images_lg, removed_cols = remove_zero_variance_columns(train_images_lg)
test_images_lg = np.delete(test_images_lg, removed_cols, 1)

##############################################################################
# Use PCA method to determine best r of the test_images_lg and train_images_lg



class GDAModel:

    def __init__(self, mu0, mu1, phi, covariance):
        self.mu0 = mu0
        self.mu1 = mu1
        self.phi = phi
        self.cov = covariance
        self.no_of_feature = mu0.shape[0]

    def predict(self, x):

        y_predict = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            if self.compute_exp_term(x[i, :], self.mu0) < self.compute_exp_term(x[i, :], self.mu1):
                y_predict[i] = 1

        return y_predict

    def compute_exp_term(self, x, mu):
        return -1 / 2 * np.dot(np.dot(np.transpose(x - mu), np.linalg.inv(self.cov)), (x - mu))

    @staticmethod
    def compute_variance(x, y, mu0, mu1):

        if y.shape[0] > 0:
            var = (np.sum((x[np.where(y == 0)[0]] - mu0) ** 2) + np.sum((x[np.where(y == 1)[0]] - mu1) ** 2))/y.shape[0]
            return var
        else:
            return None

    @staticmethod
    def gda_estimate(x, y):

        if y.shape[0] == 0:
            return GDAModel(None, None, None)
        else:
            estimate_phi = np.sum(y) / y.shape[0]

        if np.sum(y) < y.shape[0]:  # will only work if y is Bernoulli distributed
            estimate_mu0 = np.sum(x[np.where(y == 0)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu0 = None

        if np.sum(y) > 0:
            estimate_mu1 = np.sum(x[np.where(y == 1)[0], :], axis=0) / np.where(y == 0)[0].shape[0]
        else:
            estimate_mu1 = None

        # initialize covariance matrix with zeroes entries of dimension n x n
        estimate_cov = np.zeros(x.shape[1] ** 2).reshape(x.shape[1], x.shape[1])
        for i in range(x.shape[1]):
            estimate_cov[i, i] = GDAModel.compute_variance(x[:, i], y, estimate_mu0[i], estimate_mu1[i])
        model = GDAModel(estimate_mu0, estimate_mu1, estimate_phi, estimate_cov)

        return model



##############################################################################
# Try the GDA model with PCA preprocessing on the full MNIST dataset

t = np.linspace(4,620,308, dtype='int')
t = list(t)

gda_accuracy_values = np.zeros((625,1))
U_full, S_full, V_full = np.linalg.svd(train_images_lg, full_matrices=False)
for i in t:
    
    V_full = np.transpose(V_full)
    V = V_full[:,0:i]
    
    # apply PCA to the training data
    B_tilde_train = np.dot(train_images_lg, V)
    B_cross_train = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_train), B_tilde_train)), np.transpose(B_tilde_train)))
    
    # apply PCA to the test data
    B_tilde_test = np.dot(test_images_lg, V)
    B_cross_test = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_test), B_tilde_test)), np.transpose(B_tilde_test)))
    
    # Use the GDA Model
    gda_model = GDAModel.gda_estimate(B_cross_train, train_labels_lg)  # estimate
    gda_predict_labels = gda_model.predict(B_cross_test) # predict
    gda_accuracy = compute_accuracy(test_labels_lg, gda_predict_labels)  # compute accuracy
    gda_accuracy_values[i] = gda_accuracy


# Plot the data to visualize the best value of r
r = np.linspace(1,train_images_lg.shape[1],train_images_lg.shape[1], dtype='int')
r = r.reshape((train_images_lg.shape[1],1))

plt.plot(r, gda_accuracy_values)
plt.ylabel('Accuracy')
plt.xlabel('r')
plt.title('GDA Accuracy as a function of r. MNIST. (m = full)')
plt.show()




##############################################################################
# Try the GDA model with PCA preprocessing on MNIST with m = 200

train_images_short = train_images_lg[0:200,:]
train_labels_short = train_labels_lg[0:200]
test_images_short = test_images_lg[0:200,:]
test_labels_short = test_labels_lg[0:200]

gda_accuracy_short = np.zeros((train_images_short.shape[0],1))
U_full, S_full, V_full = np.linalg.svd(train_images_short, full_matrices=False)
for i in range(1,200):
    V_full = np.transpose(V_full)
    V = V_full[:,0:i]
    
    # apply PCA to the training data
    B_tilde_train = np.dot(train_images_short[:,0:V.shape[0]], V)
    B_cross_train = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_train), B_tilde_train)), np.transpose(B_tilde_train)))
    
    # apply PCA to the test data
    B_tilde_test = np.dot(test_images_short[:,0:V.shape[0]], V)
    B_cross_test = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_test), B_tilde_test)), np.transpose(B_tilde_test)))
    
    # Use the GDA Model
    gda_model = GDAModel.gda_estimate(B_cross_train, train_labels_short)  # estimate
    gda_predict_labels = gda_model.predict(B_cross_test) # predict
    gda_accuracy = compute_accuracy(test_labels_short, gda_predict_labels)  # compute accuracy
    gda_accuracy_short[i] = gda_accuracy


rr = np.linspace(1,train_images_short.shape[0],train_images_short.shape[0], dtype='int')
rr = rr.reshape((train_images_short.shape[0],1))

plt.plot(rr, gda_accuracy_short)
plt.ylabel('Accuracy')
plt.xlabel('r')
plt.title('GDA Accuracy as a function of r. MNIST. m = 200')
plt.show()

##############################################################################
# Applying PCA to simulated data with small m

np.seterr(over='ignore')

np.random.seed(1)

# parameters for the model
phi = 0.4
m = 40
m_test = 20
n = 100
mu0 = np.random.uniform(0, 7, n)
mu1 = np.random.uniform(0, 7, n)

# generate training data
y = st.bernoulli.rvs(phi, size=m)
tmp = np.random.randn(n, n)
evals = np.concatenate((100 * np.random.randn(round(n/8)), np.random.randn(n - round(n/8)))) ** 2
cov = tmp @ np.diag(evals) @ np.transpose(tmp)
x = np.zeros(m*n).reshape(m, n)
for i in range(m):
    mu = mu0 * (1 - y[i]).item() + mu1 * y[i].item()
    x[i, :] = np.random.multivariate_normal(mu, cov)

# generate test data
y_test = st.bernoulli.rvs(phi, size=m_test)

x_test = np.zeros(m_test * n).reshape(m_test, n)
for i in range(m_test):
    mu = mu0 * (1 - y_test[i]).item() + mu1 * y_test[i].item()
    x_test[i, :] = np.random.multivariate_normal(mu, cov)



# Apply PCA to the simulated data

gda_accuracy_simulated = np.zeros((x.shape[1],1))
U_full, S_full, V_full = np.linalg.svd(x, full_matrices=False)
for i in range(1,40):
    V_full = np.transpose(V_full)
    V = V_full[:,0:i]
    
    # apply PCA to the training data
    B_tilde_train = np.dot(x[:,0:V.shape[0]], V)
    B_cross_train = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_train), B_tilde_train)), np.transpose(B_tilde_train)))
    
    # apply PCA to the test data
    B_tilde_test = np.dot(x_test[:,0:V.shape[0]], V)
    B_cross_test = np.transpose(np.dot(np.linalg.inv(np.dot(np.transpose(B_tilde_test), B_tilde_test)), np.transpose(B_tilde_test)))
    
    # Use the GDA Model
    gda_model = GDAModel.gda_estimate(B_cross_train, y)  # estimate
    gda_predict_labels = gda_model.predict(B_cross_test) # predict
    gda_accuracy = compute_accuracy(y_test, gda_predict_labels)  # compute accuracy
    gda_accuracy_simulated[i] = gda_accuracy


rr = np.linspace(1,100,100, dtype='int')
rr = rr.reshape((100,1))

plt.plot(rr, gda_accuracy_simulated)
plt.ylabel('Accuracy')
plt.xlabel('r')
plt.title('GDA Accuracy as a function of r. Simulated data. m = 40')
plt.show()




