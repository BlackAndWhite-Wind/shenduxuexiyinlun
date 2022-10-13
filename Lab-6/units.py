import numpy as np

def encode_datalist(datalist):
    for i in range(len(datalist)):
        item = []
        for j in range(len(datalist[i])):
            letters = [0] * 5
            numbers = [0] * 3
            if datalist[i][j] >= 'A' and datalist[i][j] <= 'Z':
                index = ord(datalist[i][j]) - ord('A')
                letters[index] = 1
                item += letters
            elif datalist[i][j] >= '0' and datalist[i][j] <= '9':
                index = ord(datalist[i][j]) - ord('1')
                numbers[index] = 1
                item += numbers
        datalist[i] = item[:]
    return datalist

def prepare_data(data_list_encoded):
    data_prepared = []
    labels_prepared = []
    letters_len = 2 * 5
    numbers_len = 4 * 3
    for item in data_list_encoded:
        data_prepared += item[:letters_len]
        labels_prepared += item[letters_len:]
    '''
    data_prepared.shape [letters_len, data_len]
    labels_prepared.shape [numbers_len, data_len]
    '''
    data_prepared = np.array(data_prepared)
    data_prepared = data_prepared.reshape(-1, letters_len).T
    labels_prepared = np.array(labels_prepared)
    labels_prepared = labels_prepared.reshape(-1, numbers_len).T
    return data_prepared, labels_prepared

# define the sigmoid function
f = lambda s : 1 / (1 + np.exp(-s))

# derivative of sigmoid  function
df = lambda s : f(s) * (1-f(s))

def fc(w, a, x):
    # forward computing (either component or vector form)
    if x.size != 0:
        a = np.concatenate((x, a), axis=0)
    z_next = np.dot(w, a)
    a_next = f(z_next)
    return a_next, z_next

def bc(w, z, delta_next, err):
    # backward computing (you may want to take care or the `err`)
    #w = np.delete(w, 0,axis=0)
    w = np.delete(w, [0, 1, 2], axis=1)
    delta = (err + np.dot(w.T, delta_next)) * df(z)[-3:]
    return delta

# Step 4: Define Cost Function
def cost(a, y):
    J = 0
    for l in range(2, 6):
        y_out = (l - 1) * 3
        J += 1/2 * np.sum((a[l][-3:] - y[y_out - 3:y_out])**2)
    return J

# Step 5: Define Evaluation Index
def accuracy(a, y):
    acc = 0
    mini_batch = a[1].shape[1]
    for l in range(2, 6):
        y_out = (l - 1) * 3
        idx_a = np.argmax(a[l][-3:], axis=0)
        idx_y = np.argmax(y[y_out - 3:y_out], axis=0)
        acc += sum(idx_a==idx_y) / mini_batch
    return round((acc / 4), 4)
