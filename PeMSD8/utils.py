import math
import pickle
import numpy as np

def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape
    
def seq2instance(data, num_his, num_pred):
    num_step, num_sensor, dims = data.shape
    num_sample = num_step - num_his - num_pred + 1
    x = np.zeros(shape = (num_sample, num_his, num_sensor, dims))
    y = np.zeros(shape = (num_sample, num_pred, num_sensor, dims))
    for i in range(num_sample):
        x[i] = data[i : i + num_his]
        y[i] = data[i + num_his : i + num_his + num_pred]
    x = np.reshape(x, newshape = (num_sample, num_his * num_sensor, -1))
    y = np.reshape(y, newshape = (num_sample, num_pred * num_sensor, -1))
    return x, y

def loadData(args):
    '''
    (id, t) + 3-d feature (traffic flow, speed, occupancy)
    this dataset does not provide the latitude and longitude, but contains the distances between sensors
    t=0 means the time is 2016/07/01
    '''
    with open(args.data_file, 'rb') as f:
        data = pickle.load(f)
    points = data['points']
    points = np.array(points)
    num_step, num_sensor, _ = points.shape
    # train/val/test
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps
    train = points[: train_steps]
    val = points[train_steps : train_steps + val_steps]
    test = points[-test_steps :]
    # X, Y 
    trainX, trainY = seq2instance(train, args.num_his, args.num_pred)
    valX, valY = seq2instance(val, args.num_his, args.num_pred)
    testX, testY = seq2instance(test, args.num_his, args.num_pred)
    # normalization
    mean, std = np.mean(trainX[..., 2 :], axis = (0, 1)), np.std(trainX[..., 2 :], axis = (0, 1))
    trainX[..., 2 :] = (trainX[..., 2 :] - mean) / std
    valX[..., 2 :] = (valX[..., 2 :] - mean) / std
    testX[..., 2 :] = (testX[..., 2 :] - mean) / std
    # distance_matrix
    distance_matrix = data['distance_matrix'].astype(np.float32)
    return trainX, trainY, valX, valY, testX, testY, mean[0], std[0], distance_matrix
    
