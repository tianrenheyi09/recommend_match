#coding=utf-8
import pandas as pd

from preprocess import gen_data_set_sdm, gen_model_input_sdm
from sklearn.preprocessing import LabelEncoder

import sys
import os
import tensorflow as tf
from tensorflow.contrib import rnn

# from model_sdm import ModelSDM
import numpy as np

def get_inputs():
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")
    user_zip = tf.placeholder(tf.int32, [None, 1], name="user_zip")
    
    prefer_id = tf.placeholder(tf.int32, [None, 50], name="prefer_id")
    prefer_gen = tf.placeholder(tf.int32, [None, 50], name="prefer_gen")
    prefer_real_len = tf.placeholder(tf.int32, [None, 1], name="prefer_real_len")

    short_id = tf.placeholder(tf.int32, [None, 5], name="short_id")
    short_gen = tf.placeholder(tf.int32, [None, 5], name="short_gen")
    short_real_len = tf.placeholder(tf.int32, [None, 1], name="short_real_len")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    
    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name = "LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")

    return uid, user_gender, user_age, user_job, user_zip, prefer_id, prefer_gen, prefer_real_len, short_id, short_gen, short_real_len, movie_id, targets, LearningRate, dropout_keep_prob


def get_user_embedding(uid, user_gender, user_age, user_job, user_zip, feature_max_idx):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.random_uniform([feature_max_idx["user_id"], 16], -1, 1), name = "uid_embed_matrix")
        print("uid_enbed_matrix_shape:", uid_embed_matrix.get_shape().as_list(), uid.get_shape())
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name = "uid_embed_layer")
    
        gender_embed_matrix = tf.Variable(tf.random_uniform([feature_max_idx["gender"], 16], -1, 1), name= "gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name = "gender_embed_layer")
        
        age_embed_matrix = tf.Variable(tf.random_uniform([feature_max_idx["age"], 16], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")
        
        job_embed_matrix = tf.Variable(tf.random_uniform([feature_max_idx["occupation"], 16], -1, 1), name = "job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name = "job_embed_layer")

        zip_embed_matrix = tf.Variable(tf.random_uniform([feature_max_idx["zip"], 16], -1, 1), name = "zip_embed_matrix")
        zip_embed_layer = tf.nn.embedding_lookup(zip_embed_matrix, user_zip, name = "zip_embed_layer")

        ###cocnat and dense
        concat_user = tf.concat([uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer, zip_embed_layer], axis=2)
        
        user_combine = tf.layers.dense(concat_user, 32, name = "user_combine_layer", activation=tf.nn.relu)

        print("user_combine_shape:", user_combine.get_shape().as_list())

    return user_combine

def concat_attention(query, key):
    """
    :param query: [batch_size, 1, query_size] -> [batch_size, time, query_size]
    :param key:   [batch_size, time, key_size]
    :return:      [batch_size, 1, time]
        query_size should keep the same dim with key_size
    """
    # TODO: only support 1D attention at present
    # query = tf.tile(query, [1, tf.shape(key)[1], 1])
    # [batch_size, time, q_size+k_size]
    q_k = tf.concat([query, key], axis=-1)
    # [batch_size, time, 1]
    align = tf.layers.dense(q_k, 1, tf.nn.tanh)  # tf.nn.relu old
    # scale (optional)
    align = align / (key.get_shape().as_list()[-1] ** 0.5)
    align = tf.transpose(align, [0, 2, 1])
    return align

def attention(queries, keys, keys_length):
    #   '''
    # queries:     [B, H]
    # keys:        [B, T, H]
    # keys_length: [B]
    # '''
    
    hist_len = keys.get_shape()[1]
    key_masks = tf.sequence_mask(keys_length, hist_len)
    print("key_mask_shape", key_masks.get_shape().as_list())
    print("querys_shape", queries.get_shape().as_list())

    queries = tf.tile(queries, [1, hist_len, 1])  # [batch_size, T, units]
    attention_score = concat_attention(queries, keys)  # [batch_size, 1, units]

    print("atten_sscore_shape:", attention_score.get_shape().as_list())
    print("keys_shape", keys.get_shape().as_list())
    outputs = softmax_weight_sum(attention_score, keys, key_masks)
    # [batch_size, units]
    return outputs


def get_prefer_outputs(prefer_id, len_prefer_id, prefer_gen, len_prefer_gen, short_id, len_short_id, short_gen, len_short_gen, label_split, user_combine, feature_max_idx):

    # units = 32
    ##########-------longterm
    
    movie_id_emb_matrix = tf.Variable(tf.random_uniform([feature_max_idx["movie_id"], 32], -1,1), name = "movie_id_emb_matrix")
    print("movie_iemd_matrix_shape:", movie_id_emb_matrix.get_shape().as_list())

    nce_biases = tf.zeros([feature_max_idx['movie_id']], name='bias')
    
    prefer_id_embed_layer = tf.nn.embedding_lookup(movie_id_emb_matrix, prefer_id, name = "prefer_id_embed_layer")
    ###B*T*H

    # gen_emb_matrix  = tf.Variable(tf.random_uniform(feature_max_idx["genres"], 32], -1, 1), name = "gen_emb_matrix")
    gen_emb_matrix = tf.Variable(tf.random_uniform([feature_max_idx["genres"], 32], -1, 1), name = "gen_emb_matrix")
    prefer_gen_embed_layer = tf.nn.embedding_lookup(gen_emb_matrix, prefer_gen, name = "prefer_gen_embed_layer")
    ###B*T*H

    ###attention,
    print("user_combine_shape:", user_combine.get_shape().as_list())
    print("prefer_id_embed_layer_shape:", prefer_id_embed_layer.get_shape().as_list())
    print("len_prefer_id_shape", len_prefer_id.get_shape().as_list())

    prefer_id_atten = attention(user_combine, prefer_id_embed_layer, len_prefer_id)
    prefer_gen_atten = attention(user_combine, prefer_gen_embed_layer, len_prefer_gen)
    ###B*1*H
    ##concat and dense
    concat_prefer = tf.concat([prefer_id_atten, prefer_gen_atten], axis=2)
    print("concat_prefer_shape:", concat_prefer.get_shape().as_list())

    prefer_output = tf.layers.dense(concat_prefer, 32, name = "prefer_output_layer", activation=tf.nn.relu)
    print("prefer_output_shape:", prefer_output.get_shape().as_list())
    #####--------short_term
    short_id_embed_layer = tf.nn.embedding_lookup(movie_id_emb_matrix, short_id, name = "short_id_embed_layer")
    short_gen_embed_layer = tf.nn.embedding_lookup(gen_emb_matrix, short_gen, name = "short_gen_embed_layer")

    ##concat and dense
    concat_short = tf.concat([short_id_embed_layer, short_gen_embed_layer], axis = 2)
    print("concat_short_shape", concat_short.get_shape().as_list())
    short_dense_output = tf.layers.dense(concat_short, 32, name = "short_dense_output", activation=tf.nn.relu)
    print("short_dense_output", short_dense_output.get_shape().as_list())
    
    ### multi lstm
    # lstm_cell = rnn.BasicLSTMCell(num_units=32, forget_bias=1.0, state_is_tuple=True)
    # #dropout layer, output_keep_prob
    # lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=0.5)

    # # print("lstm_cell_shape:", lstm_cell.shape())
    # #MultiRNNCell LSTM
    # mlstm_cell = rnn.MultiRNNCell([lstm_cell] * 2, state_is_tuple=True)

    # # print("mlstm_cell_shape:", mlstm_cell.get_shape().as_list())
    # #init   state
    # init_state = mlstm_cell.zero_state(user_combine.get_shape().as_list()[0], dtype=tf.float32)
    # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=short_dense_output, initial_state=init_state, time_major=False)

    
    cell_list = []
    for i in range(2):
        single_cell = tf.contrib.rnn.BasicLSTMCell(32, forget_bias=1.0, state_is_tuple=True)
        cell_list.append(single_cell)
    

    final_cell = tf.contrib.rnn.MultiRNNCell(cell_list)
    print("dynamic_rnn_len_short_id", len_short_id.get_shape().as_list())
    outputs, state = tf.nn.dynamic_rnn(cell=final_cell, dtype=tf.float32, sequence_length=tf.squeeze(len_short_id, axis=1), inputs=short_dense_output)


    print("outptus_shapr:", outputs.get_shape().as_list())


    ###multi-head
    short_att_output = multiheadatten(outputs, len_short_id)

    print("multiheadatten_short_att_output_shapr:", short_att_output.get_shape().as_list())

    short_output = user_attention(user_combine, short_att_output, len_short_id)

    print("shprt_out_shape:", short_output.get_shape().as_list())
    print("prefer_output_shape", prefer_output.get_shape().as_list())
    gate_input = tf.concat([prefer_output, short_output, user_combine], axis=2)

    
    gate = tf.layers.dense(gate_input, 32, activation=tf.nn.sigmoid)
    # gate = gate_input

    print("gate_shape:", gate.get_shape().as_list())

    gate_output = tf.multiply(gate, short_output) + tf.multiply(1 - gate, prefer_output)

    print("gate_output_shape:", gate_output.get_shape().as_list())
    gate_output_reshape = tf.squeeze(gate_output, axis=1)
    print("gate_output_reshape:", gate_output_reshape.get_shape().as_list())
# outputs = tf.multiply(g, short_rep) + tf.multiply(1 - g, long_rep)
#                 tf.summary.scalar("gate", tf.reduce_mean(g))
    
    print("movie_id_emb_matrix_shape:", movie_id_emb_matrix.get_shape().as_list())
    print("label_split_shape", label_split.get_shape().as_list())
    print("gate_output_shape", gate_output.get_shape().as_list())
    print("nce_biases_shape", nce_biases.get_shape().as_list())
    # sampled_loss = tf.nn.sampled_softmax_loss(
    #     weights = movie_id_emb_matrix,
    #     biases = nce_biases,
    #     labels = label_split,
    #     inputs = gate_output_reshape, 
    #     num_sampled = 100,
    #     num_classes = feature_max_idx["movie_id"]) 
    
    # print("sample_loss_shape:", sampled_loss.get_shape().as_list())
    # print("gate_output_reshape：", tf.shape(gate_output_reshape)[0])
    # sampled_loss = tf.reshape(sampled_loss, [tf.shape(gate_output_reshape)[0], 1])
    # print("sample_loss_shape:", sampled_loss.get_shape().as_list())
    # #####sample_loss 计算方式不同

    # sampled_loss = tf.reduce_sum(sampled_loss * (tf.cast(label_split, tf.float32)))

    # return sampled_loss, gate_output_reshape, movie_id_emb_matrix
    return movie_id_emb_matrix, nce_biases, label_split, gate_output_reshape


def user_attention(user_query, keys, keys_length, use_res=True):
    """
    :param query: A 3d tensor with shape of [batch_size, T, C]
    :param keys: A 3d tensor with shape of [batch_size, T, C]
    :param key_masks: A 3d tensor with shape of [batch_size, 1]
    :return: A 3d tensor with shape of  [batch_size, 1, C]
    """
    hist_len = keys.get_shape()[1]
    key_masks = tf.sequence_mask(keys_length, hist_len)
    query = tf.layers.dense(user_query, 32, activation=tf.nn.relu)

    align = dot_attention(query, keys, True)
    
    output = softmax_weight_sum(align, keys, key_masks)

    if use_res:
        output += keys
    return tf.reduce_mean(output, 1, keep_dims=True)


def dot_attention(query, key, scale=False):
    output = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
    if scale:
        output = output/(key.get_shape().as_list()[-1] ** 0.5)
    
    return output


def multiheadatten(input_info, keys_length, num_units=32, head_num=4, scale=True, dropout_rate=0.2, future_binding=True, use_layer_norm=True,
                use_res=True,
                seed=2020):
    embedding_size = 32

    print("input_info_shap", input_info.get_shape().as_list())
    init = tf.truncated_normal_initializer(stddev=0.01, seed=seed)
    W= tf.get_variable('Q_K_V', shape=[embedding_size, num_units * 3], initializer=init)

    W_output = tf.get_variable("output_W", shape=[num_units, num_units], initializer=init)

    hist_len = input_info.get_shape()[1]
    key_masks = tf.sequence_mask(keys_length, hist_len)
    key_masks = tf.squeeze(key_masks, axis=1)
    ####input: [N, T_k, emb], key masks: [N, key_seqlen]
    Q_K_V = tf.tensordot(input_info, W, axes=(-1, 0))  # [N T_q D*3]
    # Q_K_V = tf.layers.dense(input_info, 3 * num_units)  # tf.nn.relu
    querys, keys, values = tf.split(Q_K_V, 3, -1)

    # head_num None F D
    querys = tf.concat(tf.split(querys, head_num, axis=2), axis=0)  # (h*N, T_q, C/h)
    keys = tf.concat(tf.split(keys, head_num, axis=2), axis=0)  # (h*N, T_k, C/h)
    values = tf.concat(tf.split(values, head_num, axis=2), axis=0)  # (h*N, T_k, C/h)

    # (h*N, T_q, T_k)
    align = dot_attention(querys, keys)

    key_masks = tf.tile(key_masks, [head_num, 1])  # (h*N, T_k)
    key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(input_info)[1], 1])  # (h*N, T_q, T_k)

    outputs = softmax_weight_sum(align, values, key_masks, future_binding=True)  # (h*N, T_q, C/h)
    outputs = tf.concat(tf.split(outputs, head_num, axis=0), axis=2)  # (N, T_q, C)

    outputs = tf.tensordot(outputs, W_output, axes=(-1, 0))  # (N, T_q, C)
    print("outputs_shape:", outputs.get_shape().as_list())

    # outputs = tf.layers.dense(outputs, num_units)
    # outputs = dropout(outputs, training=training)
    # if use_res:
    #     outputs += input_info
    if use_layer_norm:
        outputs = layer_norm(outputs)
    print("use_layer_norm_outputs_shape:", outputs.get_shape().as_list())
    return outputs


def layer_norm(inputs, input_axis=-1, eps=1e-9, center=True,
                scale=True):
    # input_shape = inputs.get_shape().as_list()
    # gamma = tf.get_variable("gamma", shape=input_shape[-1:], initializer=tf.ones_initializer(), trainable=True)
    # beta = tf.get_variable("beta", shape=input_shape[-1:], initializer=tf.zeros_initializer(), trainable=True)
    
    # mean = K.mean(inputs, axis=input_axis, keepdims=True)
    # variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
    # std = K.sqrt(variance + eps)
    # outputs = (inputs - mean) / std
    # if scale:
    #     outputs *= gamma
    # if center:
    #     outputs += beta


    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + eps))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable('gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable('beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta

    return outputs
    

def softmax_weight_sum(align, value, key_masks, future_binding=False):
    """
    :param align:           [batch_size, 1, T]
    :param value:           [batch_size, T, units]
    :param key_masks:       [batch_size, 1, T]
                            2nd dim size with align
    :param drop_out:
    :param future_binding:
    :return:                weighted sum vector
                            [batch_size, 1, units]
    """
    paddings = tf.ones_like(align) * (-2 ** 32 + 1)
    # print("padding_shape:", paddings.get_shape().as_list())
    # print("key_masks_shape", key_masks.get_shape().as_list())
    # print("align_shape:", align.get_shape().as_list())
    align = tf.where(key_masks, align, paddings)
    
    # print("new_lign_shape:", align.get_shape().as_list())

   
    if future_binding:
        # length = value.get_shape().as_list()[1]
        length = tf.reshape(tf.shape(value)[1], [-1])
        # lower_tri = tf.ones([length, length])
        lower_tri = tf.ones(tf.concat([length, length], axis=0))
        # [time, time]
        print("lower_tri_shape:", lower_tri.get_shape().as_list())
        try:
            lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
        except:
            lower_tri = tf.linalg.LinearOperatorLowerTriangular(lower_tri).to_dense()
        # [batch_size, time, time]
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        align = tf.where(tf.equal(masks, 0), paddings, align)

        print("masks_shape:", masks.get_shape().as_list())
        print("paddings*shape:", paddings.get_shape().as_list())
        print("align*shape:", align.get_shape().as_list())
    
    print("before_soft_align*shape:", align.get_shape().as_list())

    align = tf.nn.softmax(align)

    # align = dropout(align, training=training)
    output = tf.matmul(align, value)
    print("softmax_output_shape:", output.get_shape().as_list())

    return output


def get_batch(x, y, batch_size, index):
    start = index * batch_size
    end = (index + 1) *batch_size
    end = end if end < len(y) else len(y)

    return x[start:end], [y_ for y_ in y[start:end]]


def shuffle_data(a, b):
    res = list(zip(a,b))
    np.random.shuffle(res)
    a, b = zip(*res)


if __name__ == "__main__":
    data = pd.read_csvdata = pd.read_csv("./movielens_sample.txt")
    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zip", "genres"]
    SEQ_LEN_short = 5
    SEQ_LEN_prefer = 50

    # 1.Label Encoding for sparse features,and process sequence features with `gen_date_set` and `gen_model_input`

    features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'genres']
    feature_max_idx = {}
    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1

    user_profile = data[["user_id", "gender", "age", "occupation", "zip", "genres"]].drop_duplicates('user_id')

    item_profile = data[["movie_id"]].drop_duplicates('movie_id')

    user_profile.set_index("user_id", inplace=True)
    #
    # user_item_list = data.groupby("user_id")['movie_id'].apply(list)

    train_set, test_set = gen_data_set_sdm(data, seq_short_len=SEQ_LEN_short, seq_prefer_len=SEQ_LEN_prefer)

    train_model_input, train_label = gen_model_input_sdm(train_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)
    test_model_input, test_label = gen_model_input_sdm(test_set, user_profile, SEQ_LEN_short, SEQ_LEN_prefer)

    print("train_model_input", len(train_label), train_label)
    for key in train_model_input:
        print(key, train_model_input[key], len(train_model_input[key]))
    

    ####
tf.reset_default_graph()
train_graph = tf.Graph()
with train_graph.as_default():
    uid, user_gender, user_age, user_job,  \
                user_zip, prefer_id, prefer_gen, prefer_real_len,\
                     short_id, short_gen, short_real_len, movie_id, targets, LearningRate, dropout_keep_prob = get_inputs()
    
        
    user_combine = get_user_embedding(uid, user_gender, user_age, user_job, user_zip, feature_max_idx)

    movie_id_emb_matrix, nce_biases, label_split, gate_output_reshape = get_prefer_outputs(prefer_id, prefer_real_len, prefer_gen, prefer_real_len, short_id, short_real_len, short_gen, short_real_len, targets, user_combine, feature_max_idx)


    with tf.name_scope("loss"):
        sampled_loss = tf.nn.sampled_softmax_loss(
            weights = movie_id_emb_matrix,
            biases = nce_biases,
            labels = label_split,
            inputs = gate_output_reshape, 
            num_sampled = 100,
            num_classes = feature_max_idx["movie_id"]) 
        
        print("sample_loss_shape:", sampled_loss.get_shape().as_list())
        print("gate_output_reshape：", tf.shape(gate_output_reshape)[0])
        sampled_loss = tf.reshape(sampled_loss, [tf.shape(gate_output_reshape)[0], 1])
        print("sample_loss_shape:", sampled_loss.get_shape().as_list())
        #####sample_loss 计算方式不同

        loss = tf.reduce_sum(sampled_loss * (tf.cast(label_split, tf.float32)))
    
    # 优化损失 
#     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(0.01)
    gradients = optimizer.compute_gradients(loss)  #cost
    train_op = optimizer.apply_gradients(gradients, global_step)



with tf.Session(graph=train_graph) as sess:

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    for epoch in range(10):

        # shuffle_data(train_model_input, train_label)
        # total_batch = int(len(train_label) - 1) / batch_size  + 1

        # for i in range total_batch:
        #     x_batch, y_batch = get_batch(train_model_input, train_label)
        tmp_batch = len(train_model_input["user_id"])
        feed_dict = {
            uid: train_model_input["user_id"].reshape([tmp_batch, 1]),
            user_gender: train_model_input["gender"].reshape([tmp_batch, 1]),
            user_age: train_model_input["age"].reshape([tmp_batch, 1]),
            user_job: train_model_input["occupation"].reshape([tmp_batch, 1]),
            user_zip: train_model_input["zip"].reshape([tmp_batch, 1]),
            prefer_id: train_model_input['prefer_movie_id'],
            prefer_gen: train_model_input["prefer_genres"],
            prefer_real_len: train_model_input["prefer_sess_length"].reshape([tmp_batch, 1]),
            short_id: train_model_input["short_movie_id"],
            short_gen: train_model_input["short_genres"],
            short_real_len: train_model_input["short_sess_length"].reshape([tmp_batch, 1]),
            movie_id: train_model_input["movie_id"].reshape([tmp_batch, 1]),
            targets: train_label.reshape([tmp_batch, 1]),
            LearningRate: 0.01,
            dropout_keep_prob:0.5
        }

       
        step, train_loss, _,  item_vector, user_vector = sess.run([global_step, loss, train_op, movie_id_emb_matrix, gate_output_reshape], feed_dict=feed_dict)
        print("train_loss", train_loss, item_vector.shape, user_vector.shape)
        
        ####test-and-valid   movie_id_emb_matrix, nce_biases, label_split, gate_output_reshape
        tmp_batch = len(test_model_input["user_id"])
        feed_dict = {
            uid: test_model_input["user_id"].reshape([tmp_batch, 1]),
            user_gender: test_model_input["gender"].reshape([tmp_batch, 1]),
            user_age: test_model_input["age"].reshape([tmp_batch, 1]),
            user_job: test_model_input["occupation"].reshape([tmp_batch, 1]),
            user_zip: test_model_input["zip"].reshape([tmp_batch, 1]),
            prefer_id: test_model_input['prefer_movie_id'],
            prefer_gen: test_model_input["prefer_genres"],
            prefer_real_len: test_model_input["prefer_sess_length"].reshape([tmp_batch, 1]),
            short_id: test_model_input["short_movie_id"],
            short_gen: test_model_input["short_genres"],
            short_real_len: test_model_input["short_sess_length"].reshape([tmp_batch, 1]),
            movie_id: test_model_input["movie_id"].reshape([tmp_batch, 1]),
            targets: test_label.reshape([tmp_batch, 1]),
            LearningRate: 0.01,
            dropout_keep_prob:0.5
        }

        test_loss, test_item_vector, test_user_vector = sess.run([loss, movie_id_emb_matrix, gate_output_reshape], feed_dict=feed_dict)
        print("test_loss", test_loss, test_item_vector.shape, test_user_vector.shape)
    
    test_true_label = {line[0]: [line[3]] for line in test_set}
    embedding_dim = 32
    import numpy as np
    import faiss
    from tqdm import tqdm
    

    def recall_N(y_true, y_pred, N=50):
        return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)

    index = faiss.IndexFlatIP(embedding_dim)
    # faiss.normalize_L2(item_embs)
    index.add(item_vector)
    # faiss.normalize_L2(user_embs)
    D, I = index.search(np.ascontiguousarray(test_user_vector), 50)

    print("D_shapr:", D.shape)
    print("I_shape:", I.shape)

    s = []
    hit = 0
    for i, uid in tqdm(enumerate(test_model_input['user_id'])):
        try:
            pred = [item_profile['movie_id'].values[x] for x in I[i]]
            filter_item = None
            recall_score = recall_N(test_true_label[uid], pred, N=50)
            s.append(recall_score)
            if test_true_label[uid] in pred:
                hit += 1
        except:
            print(i)
    print("")
    print("recall", np.mean(s))
    print("hit rate", hit / len(test_model_input['user_id']))



        




    
    # embedding_dim = 32
    # # for sdm,we must provide `VarLenSparseFeat` with name "prefer_xxx" and "short_xxx" and their length
    # user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], 16),
    #                         SparseFeat("gender", feature_max_idx['gender'], 16),
    #                         SparseFeat("age", feature_max_idx['age'], 16),
    #                         SparseFeat("occupation", feature_max_idx['occupation'], 16),
    #                         SparseFeat("zip", feature_max_idx['zip'], 16),
    #                         VarLenSparseFeat(SparseFeat('short_movie_id', feature_max_idx['movie_id'], embedding_dim,
    #                                                     embedding_name="movie_id"), SEQ_LEN_short, 'mean',
    #                                          'short_sess_length'),
    #                         VarLenSparseFeat(SparseFeat('prefer_movie_id', feature_max_idx['movie_id'], embedding_dim,
    #                                                     embedding_name="movie_id"), SEQ_LEN_prefer, 'mean',
    #                                          'prefer_sess_length'),
    #                         VarLenSparseFeat(SparseFeat('short_genres', feature_max_idx['genres'], embedding_dim,
    #                                                     embedding_name="genres"), SEQ_LEN_short, 'mean',
    #                                          'short_sess_length'),
    #                         VarLenSparseFeat(SparseFeat('prefer_genres', feature_max_idx['genres'], embedding_dim,
    #                                                     embedding_name="genres"), SEQ_LEN_prefer, 'mean',
    #                                          'prefer_sess_length'),
    #                         ]

    # item_feature_columns = [SparseFeat('movie_id', feature_max_idx['movie_id'], embedding_dim)]

    # K.set_learning_phase(True)

    # import tensorflow as tf

    # if tf.__version__ >= '2.0.0':
    #     tf.compat.v1.disable_eager_execution()

    # # units must be equal to item embedding dim!
    # model = SDM(user_feature_columns, item_feature_columns, history_feature_list=['movie_id', 'genres'],
    #             units=embedding_dim, num_sampled=100, )

    # model.compile(optimizer='adam', loss=sampledsoftmaxloss)  # "binary_crossentropy")

    # history = model.fit(train_model_input, train_label,  # train_label,
    #                     batch_size=512, epochs=1, verbose=1, validation_split=0.0, )

    # K.set_learning_phase(False)
    # # 3.Define Model,train,predict and evaluate
    # test_user_model_input = test_model_input
    # all_item_model_input = {"movie_id": item_profile['movie_id'].values, }

    # user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    # item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    # # user_embs = user_embs[:, i, :]  # i in [0,k_max) if MIND
    # item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # print(user_embs.shape)
    # print(item_embs.shape)

    # # test_true_label = {line[0]: [line[3]] for line in test_set}
    # #
    # # import numpy as np
    # # import faiss
    # # from tqdm import tqdm
    # # from deepmatch.utils import recall_N
    # #
    # # index = faiss.IndexFlatIP(embedding_dim)
    # # # faiss.normalize_L2(item_embs)
    # # index.add(item_embs)
    # # # faiss.normalize_L2(user_embs)
    # # D, I = index.search(np.ascontiguousarray(user_embs), 50)
    # # s = []
    # # hit = 0
    # # for i, uid in tqdm(enumerate(test_user_model_input['user_id'])):
    # #     try:
    # #         pred = [item_profile['movie_id'].values[x] for x in I[i]]
    # #         filter_item = None
    # #         recall_score = recall_N(test_true_label[uid], pred, N=50)
    # #         s.append(recall_score)
    # #         if test_true_label[uid] in pred:
    # #             hit += 1
    # #     except:
    # #         print(i)
    # # print("")
    # # print("recall", np.mean(s))
    # # print("hit rate", hit / len(test_user_model_input['user_id']))
