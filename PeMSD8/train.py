import math
import argparse
import utils, model
import time, datetime
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--time_slot', type = int, default = 5 * 60, help = 'time slot (5 mins)')
parser.add_argument('--num_his', type = int, default = 12, help = 'number of historical time steps')
parser.add_argument('--num_pred', type = int, default = 12, help = 'number of prediction time steps')
parser.add_argument('--num_neighbor', type = int, default = 6, help = 'number of neighboring points')
parser.add_argument('--rho', type = int, default = 1000, help = 'the threshold of distance(miles) in conv-inter')
parser.add_argument('--D1', type = int, default = 64, help = 'the dimension in the first convolution module')
parser.add_argument('--D2', type = int, default = 128, help = 'the dimension in the second convolution module')
parser.add_argument('--train_ratio', type = float, default = 0.7, help = 'train/val/test')
parser.add_argument('--val_ratio', type = float, default = 0.1, help = 'train/val/test')
parser.add_argument('--test_ratio', type = float, default = 0.2, help = 'train/val/test')
parser.add_argument('--batch_size', type = int, default = 16, help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 1000, help = 'number of epoches to run')
parser.add_argument('--patience', type = int, default = 10, help = 'patience to stop training')
parser.add_argument('--learning_rate', type=float, default = 0.001, help = 'initial learning rate')
parser.add_argument('--data_file', default = './data/PeMSD8.pickle', help = 'data file')
parser.add_argument('--model_file', default = './data/STPC-Net', help = 'save the model to disk')
parser.add_argument('--log_file', default = './data/log', help = 'log file')
args = parser.parse_args()

start = time.time()
log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
trainX, trainY, valX, valY, testX, testY, mean, std, distance_matrix = utils.loadData(args)
utils.log_string(log, 'trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')

# train model
utils.log_string(log, 'compiling model...')
num_train, N, in_dims = trainX.shape
N1 = trainY.shape[1]
x, y_coord, label, is_training = model.placeholder(in_dims, N, N1)
pred = model.STPC_Net(
    x, y_coord, D1 = args.D1, D2 = args.D2, distance_matrix = distance_matrix, num_neighbor = args.num_neighbor, 
    rho = args.rho, time_slot = args.time_slot, num_his = args.num_his, num_pred = args.num_pred, is_training = is_training)
pred = pred * std + mean
loss = model.loss(pred, label)
tf.compat.v1.add_to_collection('pred', pred)
tf.compat.v1.add_to_collection('loss', loss)
global_step = tf.Variable(0, trainable = False)
optimizer = tf.compat.v1.train.AdamOptimizer(args.learning_rate)
train_op = optimizer.minimize(loss, global_step = global_step)
parameters = 0
for variable in tf.compat.v1.trainable_variables():
    parameters += np.product([x.value for x in variable.get_shape()])
utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
utils.log_string(log, 'model compiled!')
saver = tf.compat.v1.train.Saver()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config = config)
sess.run(tf.compat.v1.global_variables_initializer())
utils.log_string(log, '**** training model ****')
num_val = valX.shape[0]
wait = 0
val_loss_min = np.inf
for epoch in range(args.max_epoch):
    if wait >= args.patience:
        utils.log_string(log, 'early stop at epoch: %04d' % (epoch))
        break
    # shuffle
    permutation = np.random.permutation(num_train)
    trainX = trainX[permutation]
    trainY = trainY[permutation]
    # train loss
    trainPred = []
    num_batch = math.ceil(num_train / args.batch_size)
    start_train = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            x: trainX[start_idx : end_idx],
            y_coord: trainY[start_idx : end_idx, :, : 2],
            label: trainY[start_idx : end_idx, :, 2],
            is_training: True}
        _, pred_batch = sess.run([train_op, pred], feed_dict = feed_dict)
        trainPred.append(pred_batch)
    end_train = time.time()
    trainPred = np.concatenate(trainPred, axis = 0)
    train_loss = utils.metric(trainPred, trainY[..., 2])[0]
    # val loss
    valPred = []
    num_batch = math.ceil(num_val / args.batch_size)
    start_val = time.time()
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            x: valX[start_idx : end_idx],
            y_coord: valY[start_idx : end_idx, :, : 2],
            is_training: False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        valPred.append(pred_batch)
    end_val = time.time()
    valPred = np.concatenate(valPred, axis = 0)
    val_loss = utils.metric(valPred, valY[..., 2])[0]
    utils.log_string(log, '%s | epoch: %04d/%d, training time: %.1fs, inference time: %.1fs, train loss: %.4f, val_loss: %.4f' %
                     (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch + 1, args.max_epoch, end_train - start_train, end_val - start_val, train_loss, val_loss))
    if val_loss <= val_loss_min:
        utils.log_string(log, 'val loss decrease from %.4f to %.4f, saving model to %s' % (val_loss_min, val_loss, args.model_file))
        wait = 0
        val_loss_min = val_loss
        saver.save(sess, args.model_file)
    else:
        wait += 1
        
# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
saver.restore(sess, args.model_file)
utils.log_string(log, 'model restored!')
utils.log_string(log, 'evaluating...')
num_test = testX.shape[0]
trainPred = []
num_batch = math.ceil(num_train / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        x: trainX[start_idx : end_idx],
        y_coord: trainY[start_idx : end_idx, :, : 2],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    trainPred.append(pred_batch)
trainPred = np.concatenate(trainPred, axis = 0)
valPred = []
num_batch = math.ceil(num_val / args.batch_size)
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        x: valX[start_idx : end_idx],
        y_coord: valY[start_idx : end_idx, :, : 2],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    valPred.append(pred_batch)
valPred = np.concatenate(valPred, axis = 0)
testPred = []
num_batch = math.ceil(num_test / args.batch_size)
start_test = time.time()
for batch_idx in range(num_batch):
    start_idx = batch_idx * args.batch_size
    end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
    feed_dict = {
        x: testX[start_idx : end_idx],
        y_coord: testY[start_idx : end_idx, :, : 2],
        is_training: False}
    pred_batch = sess.run(pred, feed_dict = feed_dict)
    testPred.append(pred_batch)
end_test = time.time()
testPred = np.concatenate(testPred, axis = 0)
train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY[..., 2])
val_mae, val_rmse, val_mape = utils.metric(valPred, valY[..., 2])
test_mae, test_rmse, test_mape = utils.metric(testPred, testY[..., 2])
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                MAE\t\tRMSE\t\tMAPE')
utils.log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' % (train_mae, train_rmse, train_mape * 100))
utils.log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' % (val_mae, val_rmse, val_mape * 100))
utils.log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' % (test_mae, test_rmse, test_mape * 100))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
sess.close()