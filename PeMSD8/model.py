import math
import tf_utils
import tensorflow as tf

def placeholder(in_dims, N, N1):
    x = tf.compat.v1.placeholder(shape = (None, N, in_dims), dtype = tf.float32, name = 'x')
    y_coord = tf.compat.v1.placeholder(shape = (None, N1, 2), dtype = tf.float32, name = 'y_coord')
    label = tf.compat.v1.placeholder(shape = (None, N1), dtype = tf.float32, name = 'label')
    is_training = tf.compat.v1.placeholder(shape = (), dtype = tf.bool, name = 'is_training')
    return x, y_coord, label, is_training
    
def compute_delta_t(source, target):
    '''
    t_target - t_source
    source: [batch_size, num_source, 1] # t
    target: [batch_size, num_target, 1] 
    return: [batch_size, num_target, num_souce]
    '''
    return target - tf.transpose(source, perm = (0, 2, 1))

def look_up_distance(source, target, distance_matrix):
    '''
    compute the distance as a look-up table operation
    distance_matrix is the pre-calculated distance matrix, matrix[i, j] denotes the distance from sensor i to sensor j
    return d, which denotes the distance from target to source
    source:          [batch_size, num_source, 1] # id
    target:          [batch_size, num_target, 1]
    distance_matrix: [batch_size, num_sensor, num_sensor]
    return:          [batch_size, num_target, num_source]
    '''
    batch_size = tf.shape(source)[0]
    num_source = source.get_shape()[1].value
    num_target = target.get_shape()[1].value
    source = tf.tile(source, multiples = (1, num_target, 1))
    source = tf.reshape(source, shape = (-1, 1))
    target = tf.tile(target, multiples = (1, 1, num_source))
    target = tf.reshape(target, shape = (-1, 1))
    indices = tf.concat((target, source), axis = -1) # target to source
    indices = tf.cast(indices, dtype = tf.int32)
    distance = tf.gather_nd(distance_matrix, indices = indices)
    distance = tf.reshape(distance, shape = (batch_size, num_target, num_source))
    return distance
    
def mlp(x, units, activations, is_training):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        x = tf_utils.conv2d(
            x, output_dims = num_unit, kernel_size = [1, 1], stride = [1, 1],
            padding = 'VALID', use_bias = True, activation = activation,
            bn = False, bn_decay = None, is_training = is_training)
    return x

def conv_intra(source, target, dim, num_neighbor, epsilon, is_training):
    '''
    conv-intra operator
    source: [batch_size, num_source, 1, 2 + d] (id, t)
    target: [batch_size, num_target, 1, 2] 
    return: [batch_size, num_target, 1, dim]
    '''
    inf = 2 ** 15 - 1
    eps = 1e-5
    batch_size = tf.shape(source)[0]
    num_source = source.get_shape()[1].value
    num_target = target.get_shape()[1].value

    # search the neighborhood
    # delta_t: [batch_size, num_target, num_source]
    id_source, id_target = source[..., 0], target[..., 0]
    if_intra = tf.equal(id_target - tf.transpose(id_source, perm = (0, 2, 1)), 0)
    delta_t = compute_delta_t(source[..., 1], target[..., 1])
    if_neighbor = tf.logical_and(delta_t >= 0, delta_t <= epsilon)
    delta_t = tf.compat.v2.where(condition = tf.logical_and(if_intra, if_neighbor), x = delta_t, y = inf) # mask
    # top k
    # delta_t, indices: [batch_size, num_target, num_neighbor]
    delta_t, indices = tf.nn.top_k(-delta_t, k = num_neighbor)
    delta_t = -delta_t
    indices = tf.compat.v2.where(condition = delta_t <= epsilon, x = indices, y = num_source)
    # outputs: [batch_size, num_target, num_neighbor, 2 + d]
    batch_indices = tf.reshape(tf.range(batch_size), shape = (-1, 1, 1, 1)) 
    batch_indices = tf.tile(batch_indices, multiples = (1, num_target, num_neighbor, 1))
    indices = tf.expand_dims(indices, axis = -1)
    indices = tf.concat((batch_indices, indices), axis = -1)
    source = tf.pad(source, paddings = [[0, 0], [0, 1], [0, 0], [0, 0]], mode = 'CONSTANT')
    outputs = tf.gather_nd(tf.squeeze(source, axis = 2), indices = indices)

    # feature projection
    # outputs: [batch_size, num_target, num_neighbor, dim]
    coordinate = tf.subtract(outputs[..., 1 : 2], target[..., 1 : 2])
    delta_t = tf.expand_dims(delta_t, axis = -1)
    coordinate = tf.compat.v2.where(condition = delta_t <= epsilon, x = coordinate, y =  0)  
    maximum = tf.reduce_max(coordinate, axis = (1, 2), keepdims = True)
    minimum = tf.reduce_min(coordinate, axis = (1, 2), keepdims = True)
    coordinate = (coordinate - minimum) / (maximum - minimum + eps) # normalization
    coordinate = mlp(coordinate, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    outputs = tf.concat((coordinate, outputs[..., 2 :]), axis = -1)
    outputs = mlp(outputs, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    
    # features aggregation
    maximum = tf.reduce_max(delta_t, axis = (2, 3), keepdims = True)
    minimum = tf.reduce_min(delta_t, axis = (2, 3), keepdims = True)
    weight = (delta_t - minimum) / (maximum - minimum + eps) # normalization for every output point 
    weight = mlp(weight, units = [dim, dim, 1], activations = [tf.nn.relu, tf.nn.relu, None], is_training = is_training)
    outputs = tf.multiply(outputs, weight)
    outputs = tf.reduce_sum(outputs, axis = 2, keepdims = True)
    outputs = mlp(outputs, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    return outputs

def conv_inter(source, target, dim, distance_matrix, num_neighbor, theta, rho, is_training):
    '''
    conv-inter operator
    source: [batch_size, num_source, 1, 2 + d] (id, t)
    target: [batch_size, num_target, 1, 2] 
    return: [batch_size, num_target, 1, dim]
    '''
    inf = 2 ** 15 - 1
    eps = 1e-5
    batch_size = tf.shape(source)[0]
    num_source = source.get_shape()[1].value
    num_target = target.get_shape()[1].value
    
    # search the neighborhood
    # distance: [batch_size, num_target, num_source]
    id_source, id_target = source[..., 0], target[..., 0]
    if_inter = tf.not_equal(id_target - tf.transpose(id_source, perm = (0, 2, 1)), 0)
    delta_t = tf.abs(compute_delta_t(source[..., 1], target[..., 1]))
    distance = look_up_distance(id_source, id_target, distance_matrix)
    if_neighbor = tf.logical_and(distance <= rho, delta_t <= theta)
    distance = tf.compat.v2.where(condition = tf.logical_and(if_inter, if_neighbor), x = distance, y = inf) # mask
    # top k
    # distance, indices: [batch_size, num_target, num_neighbor]
    distance, indices = tf.nn.top_k(-distance, k = num_neighbor)
    distance = -distance
    indices = tf.compat.v2.where(condition = distance <= rho, x = indices, y = num_source)
    # outputs: [batch_size, num_target, num_neighbor, 2 + d]
    batch_indices = tf.reshape(tf.range(batch_size), shape = (-1, 1, 1, 1)) 
    batch_indices = tf.tile(batch_indices, multiples = (1, num_target, num_neighbor, 1))
    indices = tf.expand_dims(indices, axis = -1)
    indices = tf.concat((batch_indices, indices), axis = -1)
    source = tf.pad(source, paddings = [[0, 0], [0, 1], [0, 0], [0, 0]], mode = 'CONSTANT')
    outputs = tf.gather_nd(tf.squeeze(source, axis = 2), indices = indices)
    
    # feature projection
    # outputs: [batch_size, num_target, num_neighbor, dim]
    coordinate = tf.subtract(outputs[..., 1 : 2], target[..., 1 : 2])
    distance = tf.expand_dims(distance, axis = -1)
    coordinate = tf.compat.v2.where(condition = distance <= rho, x = coordinate, y = 0)
    maximum = tf.reduce_max(coordinate, axis = (1, 2), keepdims = True)
    minimum = tf.reduce_min(coordinate, axis = (1, 2), keepdims = True)
    coordinate = (coordinate - minimum) / (maximum - minimum + eps) # normalization
    coordinate = mlp(coordinate, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    outputs = tf.concat((coordinate, outputs[..., 2 :]), axis = -1)    
    outputs = mlp(outputs, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    
    # features aggregation
    maximum = tf.reduce_max(distance, axis = (2, 3), keepdims = True)
    minimum = tf.reduce_min(distance, axis = (2, 3), keepdims = True)
    weight = (distance - minimum) / (maximum - minimum + eps) # normalization for every output point   
    weight = mlp(weight, units = [dim, dim, 1], activations = [tf.nn.relu, tf.nn.relu, None], is_training = is_training)
    outputs = tf.multiply(outputs, weight)
    outputs = tf.reduce_sum(outputs, axis = 2, keepdims = True)
    outputs = mlp(outputs, units = [dim, dim], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    return outputs

def fusion(intra, inter, dim, is_training):
    '''
    gated fusion
    z = sigmoid(f1(x1) + f2(x2))
    y = z * x1 + (1 - z) * x2
    '''
    x1 = mlp(intra, units = dim, activations = None, is_training = is_training)
    x2 = mlp(inter, units = dim, activations = None, is_training = is_training)  
    z = tf.nn.sigmoid(tf.add(x1, x2)) 
    outputs = tf.add(tf.multiply(z, intra), tf.multiply(1 - z, inter))
    outputs = mlp(outputs, units = dim, activations = tf.nn.relu, is_training = is_training)
    return outputs

def STPC_Net(x, y_coord, D1, D2, distance_matrix, num_neighbor, rho, time_slot, num_his, num_pred, is_training):
    '''
    x -> mlp1 -> conv1 -> mlp2 -> conv2 -> combination -> mlp3 -> pred
    x:       [batch_size, N, 2 + d]
    y_coord: [batch_size, N1, 2]
    return:  [batch_size, N1]
    '''
    N1 = y_coord.get_shape()[1].value
    # input
    x = tf.expand_dims(x, axis = 2)
    x_coord = x[..., : 2]
    x_feature = x[..., 2 :]
    y_coord = tf.expand_dims(y_coord, axis = 2)
    # mlp1
    x_feature = mlp(x_feature, units = [D1, D1], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)
    # conv1
    x = tf.concat((x_coord, x_feature), axis = -1)
    x_intra = conv_intra(x, x_coord, dim = D1, num_neighbor = num_neighbor, epsilon = time_slot * num_his, is_training = is_training)
    x_intra = tf.add(x_feature, x_intra)
    x_inter = conv_inter(
        x, x_coord, dim = D1, distance_matrix = distance_matrix, num_neighbor = num_neighbor,
        theta = time_slot, rho = rho, is_training = is_training)   
    x_inter = tf.add(x_feature, x_inter)
    x_feature = fusion(x_intra, x_inter, dim = D1, is_training = is_training)
    # mlp2
    x_feature = mlp(x_feature, units = [D1, D2], activations = [tf.nn.relu, tf.nn.relu], is_training = is_training)    
    # conv2
    x = tf.concat((x_coord, x_feature), axis = -1)
    y_intra = conv_intra(x, y_coord, dim = D2, num_neighbor = num_his, epsilon = time_slot * (num_his + num_pred), is_training = is_training)
    y_inter = conv_inter(
        x, y_coord, dim = D2, distance_matrix = distance_matrix, num_neighbor = num_neighbor,
        theta = time_slot * (num_his + num_pred), rho = rho, is_training = is_training)
    y_feature = fusion(y_intra, y_inter, dim = D2, is_training = is_training)    
    # combination 
    global_feature = tf_utils.avg_pool2d(y_feature, kernel_size = [N1, 1], stride = [1, 1], padding = 'VALID')
    global_feature = tf.tile(global_feature, multiples = (1, N1, 1, 1))
    y = tf.concat((y_feature, global_feature), axis = -1)
    # mlp3
    y = mlp(y, units = [D2, D1, 1], activations = [tf.nn.relu, tf.nn.relu, None], is_training = is_training)
    return tf.squeeze(y, axis = (2, 3))

def loss(pred, label):
##    return tf.losses.absolute_difference(label, pred) # MAE
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss
