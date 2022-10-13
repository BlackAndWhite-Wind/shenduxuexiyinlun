import matplotlib.pyplot as plt
import numpy as np
import math, pickle
from units import encode_datalist, prepare_data, fc, bc, cost, accuracy

# prepare the data set
train_list = [
    'AA1212', 'AC1231', 'AD1221', 'AE1213',
    'BA2312', 'BB2323', 'BC2331', 'BE2313',
    'CB3123', 'CC3131', 'CD3121', 'CE3113',
    'DA2112', 'DB2123', 'DC2131', 'DD2121',
    'EA1312', 'EB1323', 'ED1321', 'EE1313'
]
test_list = [
    'AB1223', 'BD2321', 'CA3112', 'DE2113', 'EC1331'
]

# encode datasets
train_list_encoded = encode_datalist(train_list)
test_list_encoded = encode_datalist(test_list)

train_size = 20
# prepare training data
trainData, trainLabels = prepare_data(train_list)
print(trainData.shape, trainLabels.shape)

# prepare testing data
testData, testLabels = prepare_data(test_list)
print(testData.shape, testLabels.shape)

# choose parameters
L = 5
# define the network architecture
layer_size = [10, 6, 6, 6, 6]

# initialize weights
w = {}
for l in range(1, L):
    w[l] = 2 * np.random.randn(layer_size[l] - 3, layer_size[l-1])
alpha = 0.1  # initialize learning rate

# train
J = []
Acc = []
max_epoch = 500 # number of training epoch 20
mini_batch = 4 # number of sample of each mini batch 4
# loop until converge
for i in range(max_epoch):
    idxs = np.random.permutation(train_size)
    # for each mini-batch
    for k in range(math.ceil(train_size / mini_batch)):
        start_idx = k*mini_batch 
        end_idx = min((k+1)*mini_batch, train_size)
        
        a, z, delta = {}, {}, {}
        batch_indices = idxs[start_idx:end_idx]
        a[1] = trainData[:, batch_indices]
        y = trainLabels[:, batch_indices]
        # forward computation
        for l in range(1, L):
            y_out = (l - 1) * 3
            a[l+1], z[l+1] = fc(w[l], a[l], y[y_out - 3:y_out])

        delta[L] = (a[L][-3:] - y[-3:]) * (a[L]*(1-a[L]))[-3:]
        # backward computation (need some attention here)
        for l in range(L-1, 1, -1):
            y_out = (l - 1) * 3
            delta[l] = bc(w[l], z[l], delta[l+1], a[l][-3:] - y[y_out - 3:y_out])
        
        # update weight
        for l in range(1, L):
            #delat_sup_list = [0] * 4
            #delta_sup = np.array(delat_sup_list).reshape(1, 4)
            #delta_sup = np.concatenate((delta_sup, delta[l+1]), axis=0)

            y_out = (l - 1) * 3
            a_l_sup = np.concatenate((y[y_out - 3:y_out], a[l]), axis=0)

            grad_w = np.dot(delta[l+1], a_l_sup.T)
            w[l] = w[l] - alpha*grad_w

        # cost function on train batch (sums from all layers)
        batch_J = cost(a, y)/mini_batch
        J.append(batch_J)

        # accuary on train batch
        batch_Acc = accuracy(a, y)
        Acc.append(batch_Acc)
    
    # optionally you can display J and Acc on-the-fly
    # plot(J,'-b')
    # drawnow
    if i % 50 == 0:
        a[1] = testData
        y = testLabels
        # forward computation 
        for l in range(1, L):
            y_out = (l - 1) * 3
            a[l+1], z[l+1] = fc(w[l], a[l], y[y_out - 3:y_out])
        print(i, "training acc:", Acc[-1], 'test acc:', accuracy(a, y))


# save model
plt.figure()
plt.plot(J)
plt.savefig("J.png")
plt.close()
plt.figure()               
plt.plot(Acc)
plt.savefig("Acc.png")
plt.close()
# Step 8: Store the Network Parameters
# save model
model_name = 'model.pkl'
with open(model_name, 'wb') as f:
    pickle.dump([w, layer_size], f)
print("model saved to {}".format(model_name))
