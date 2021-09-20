import pickle
import numpy as np
np.random.seed(0)  # seed

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)
    
def metric(pred, label):
    '''
    pred:   [batch_size, num_point, num_class]
    label:  [batch_size, num_point, num_class]
    return: average precision, recall, F-measure, accuracy
    '''
    batch_size, num_point, num_class = pred.shape
    pred = np.argmax(pred, axis = 2)
    label = np.argmax(label, axis = 2)
    Precision = []
    Recall = []
    F1 = []
    accuracy = np.sum(pred == label) / (batch_size * num_point)
    for i in range(num_class):
        correct = np.sum(np.logical_and(pred == i, label == i))
        precision = correct / np.sum(pred == i)
        recall = correct / np.sum(label == i)
        f1 = 2 * precision * recall / (precision + recall)
        Precision.append(precision)
        Recall.append(recall)
        F1.append(f1)
    return 100 * accuracy, 100 * np.mean(Precision), 100 * np.mean(Recall), 100 * np.mean(F1)

def loadData(args):
    '''
    0: walk
    1: bike
    2: bus
    3: drive
    4: train
    '''
    with open(args.data_file, 'rb') as f:
        points = pickle.load(f)
    total_points = len(points)
    num_sample = total_points // args.num_point
    total_points = num_sample * args.num_point
    points = points[: total_points]
    # X, Y
    # t = 0 -> 2007.04.12 10:21:16
    X = np.zeros(shape = (total_points, 4 + 6)) # (id, lat, lng, t) + 6-d feature
    Y = np.zeros(shape = (total_points, 5))
    for i in range(total_points):
        X[i] = points[i][: -1]
        Y[i, points[i][-1]] = 1
    X = np.reshape(X, newshape = (num_sample, args.num_point, -1))
    Y = np.reshape(Y, newshape = (num_sample, args.num_point, -1))
    # shuffle
    permutation = np.random.permutation(num_sample)
    X = X[permutation]
    Y = Y[permutation] 
    # train/val/test
    num_train = round(args.train_ratio * num_sample)
    num_test = round(args.test_ratio * num_sample)
    num_val = num_sample - num_train - num_test
    trainX = X[: num_train]
    valX = X[num_train : num_train + num_val]
    testX = X[-num_test :]
    trainY = Y[: num_train]
    valY = Y[num_train : num_train + num_val]
    testY = Y[-num_test :]    
    # normalization
    mean = np.mean(trainX[..., 4 :], axis = (0, 1), keepdims = True)
    std = np.std(trainX[..., 4 :], axis = (0, 1), keepdims = True)
    trainX[..., 4 :] = (trainX[..., 4 :] - mean) / std
    valX[..., 4 :] = (valX[..., 4 :] - mean) / std
    testX[..., 4 :] = (testX[..., 4 :] - mean) / std
    return trainX, trainY, valX, valY, testX, testY    
