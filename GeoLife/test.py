import math
import time
import utils
import argparse
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type = int, default = 512, help = 'number of point in each sample')
parser.add_argument('--train_ratio', type = float, default = 0.7, help = 'train/val/test')
parser.add_argument('--val_ratio', type = float, default = 0.1, help = 'train/val/test')
parser.add_argument('--test_ratio', type = float, default = 0.2, help = 'train/val/test')
parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
parser.add_argument('--data_file', default = './data/GeoLife.pickle', help = 'data file')
parser.add_argument('--model_file', default = './data/STPC-Net', help = 'save the model to disk')
parser.add_argument('--log_file', default = './data/log', help = 'log file')
args = parser.parse_args()

start = time.time()
log = open(args.log_file, 'w')
utils.log_string(log, str(args)[10 : -1])

# load data
utils.log_string(log, 'loading data...')
trainX, trainY, valX, valY, testX, testY = utils.loadData(args)
utils.log_string(log, 'trainX: %s\t\ttrainY: %s' % (trainX.shape, trainY.shape))
utils.log_string(log, 'valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
utils.log_string(log, 'testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
utils.log_string(log, 'data loaded!')
        
# test model
utils.log_string(log, '**** testing model ****')
utils.log_string(log, 'loading model from %s' % args.model_file)
graph = tf.Graph()
with graph.as_default():
    saver = tf.compat.v1.train.import_meta_graph(args.model_file + '.meta')
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with tf.compat.v1.Session(graph = graph, config = config) as sess:
    saver.restore(sess, args.model_file)
    parameters = 0
    for variable in tf.compat.v1.trainable_variables():
        parameters += np.product([x.value for x in variable.get_shape()])
    utils.log_string(log, 'trainable parameters: {:,}'.format(parameters))
    pred = graph.get_collection(name = 'pred')[0]
    utils.log_string(log, 'model restored!')
    utils.log_string(log, 'evaluating...')
    num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
    trainPred = []
    num_batch = math.ceil(num_train / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_train, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': trainX[start_idx : end_idx],
            'Placeholder_2:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        trainPred.append(pred_batch)
    trainPred = np.concatenate(trainPred, axis = 0)
    valPred = []
    num_batch = math.ceil(num_val / args.batch_size)
    for batch_idx in range(num_batch):
        start_idx = batch_idx * args.batch_size
        end_idx = min(num_val, (batch_idx + 1) * args.batch_size)
        feed_dict = {
            'Placeholder:0': valX[start_idx : end_idx],
            'Placeholder_2:0': False}
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
            'Placeholder:0': testX[start_idx : end_idx],
            'Placeholder_2:0': False}
        pred_batch = sess.run(pred, feed_dict = feed_dict)
        testPred.append(pred_batch)
    end_test = time.time()
    testPred = np.concatenate(testPred, axis = 0)
train_acc, train_prec, train_rec, train_F1 = utils.metric(trainPred, trainY)
val_acc, val_prec, val_rec, val_F1 = utils.metric(valPred, valY)
test_acc, test_prec, test_rec, test_F1 = utils.metric(testPred, testY)
utils.log_string(log, 'testing time: %.1fs' % (end_test - start_test))
utils.log_string(log, '                Accuracy\tPrecision\tRecall\t\tF1')
utils.log_string(log, 'train           %.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%' % (train_acc, train_prec, train_rec, train_F1))
utils.log_string(log, 'val             %.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%' % (val_acc, val_prec, val_rec, val_F1))
utils.log_string(log, 'test            %.2f%%\t\t%.2f%%\t\t%.2f%%\t\t%.2f%%' % (test_acc, test_prec, test_rec, test_F1))
end = time.time()
utils.log_string(log, 'total time: %.1fmin' % ((end - start) / 60))
log.close()
sess.close()
