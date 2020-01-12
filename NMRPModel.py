import tensorflow as tf
from rnn_cell import create_single_cell
from Configure import AttentionNetworkConfiguration
import numpy as np
from  deepctr.models import AFM

class AttentionNetwork(object):
    def __init__(self, name, configuration, training):
        self.name = name
        self.configuration = configuration
        self.training = training

        # document_size*固定的序列长度  里面保存的其实就是每个单词的位置
        self.review_documents = tf.placeholder(dtype=tf.int32, shape=(None, self.configuration.sequence_length))
        self.review_lengths = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.review_masks = tf.placeholder(dtype=tf.float32, shape=(None, self.configuration.sequence_length))
        # 预训练中的嵌入单词个数*每个单词嵌入的维度
        self.pretrained_embeddings = tf.placeholder(dtype=tf.float32, shape=(self.configuration.vocabulary_size, self.configuration.m))
        # reuse=false 创建变量   ==true 获取变量
        with tf.variable_scope(name_or_scope="{}_word_embedding".format(self.name), reuse=not self.training):
            with tf.device(device_name_or_function="/cpu:0"):
                # 嵌入的所有单词向量初始化 Word Embedding Layer
                self.word_embeddings = tf.get_variable(name="_word_embeddings", shape=(self.configuration.vocabulary_size, self.configuration.m), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01), trainable=True)
                # 单词评论嵌入 None(batch_size)*sequence_length*word_size
                self.embedded_reviews = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.review_documents)
                #self.assign_embeddings = tf.assign(ref=self.word_embeddings, value=self.pretrained_embeddings)

        # BIGRU Encoding Layer
        with tf.variable_scope(name_or_scope="{}_sequence_encoding".format(self.name), reuse=not self.training):
            self.cell_forward = create_single_cell(unit_type=self.configuration.unit_type, num_units=int(self.configuration.s / 2), dropout=self.configuration.dropout, training=self.training)
            self.cell_backward = create_single_cell(unit_type=self.configuration.unit_type, num_units=int(self.configuration.s / 2), dropout=self.configuration.dropout, training=self.training)

            # sequence_length 输入序列的实际长度（可选，默认为输入序列的最大长度）
            self.rnn_outputs, self.rnn_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self.cell_forward, cell_bw=self.cell_backward, inputs=self.embedded_reviews, sequence_length=self.review_lengths, dtype=tf.float32, swap_memory=True)

            # word_annotations: [batch_size * document_size, sequence_length, s]
            # 把两个单向的lstm 结果联合在一起
            self.word_annotations = tf.concat(values=self.rnn_outputs, axis=2)

            # reshaped_annotations: [batch_size * document_size * sequence_length, s]
            self.reshaped_annotations = tf.reshape(tensor=self.word_annotations, shape=(-1, self.configuration.s))

        self.predicted_features = []
        self.predicted_featuresu=[]

        # Topical Attention Layer　　　　　
        with tf.variable_scope(name_or_scope="{}_topical_attention".format(self.name), reuse=not self.training):
            self.transformation_weights = [tf.get_variable(name="transformation_weights_{}".format(i), shape=(self.configuration.s, self.configuration.dim), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) for i in range(self.configuration.K)]
            self.transformation_biases = [tf.get_variable(name="transformation_biases_{}".format(i), shape=(self.configuration.dim,), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) for i in range(self.configuration.K)]

            self.contextual_weights = [tf.get_variable(name="contextual_weights_{}".format(i), shape=(self.configuration.dim, 1), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) for i in range(self.configuration.K)]
            self.projection_weights = [tf.get_variable(name="projection_weights_{}".format(i), shape=(self.configuration.s, 1), dtype=tf.float32, initializer=tf.random_uniform_initializer(minval=-0.01, maxval=0.01)) for i in range(self.configuration.K)]

            for i in range(self.configuration.K):
                # transformed_annotations: [batch_size * document_size * sequence_length, dim]
                self.transformed_annotations = tf.tanh(x=tf.nn.xw_plus_b(x=self.reshaped_annotations, weights=self.transformation_weights[i], biases=self.transformation_biases[i]))

                # attention_activations: [batch_size * document_size, sequence_length]
                self.attention_activations = tf.reshape(tensor=tf.exp(x=tf.matmul(a=self.transformed_annotations, b=self.contextual_weights[i])), shape=[-1, self.configuration.sequence_length])
                # attention_activations: [batch_size * document_size, sequence_length]   review_masks非常像是 去掉常用词  把常用词变成 0  其它词为1
                self.attention_activations = tf.multiply(x=self.attention_activations, y=self.review_masks)

                # attention_distributions: [batch_size * document_size, sequence_length, 1]　　　[batch_size * document_size , sequence_length]
                self.attention_distributions = tf.reshape(tensor=tf.div(x=self.attention_activations, y=tf.reduce_sum(input_tensor=self.attention_activations,
                        axis=1, keep_dims=True)+0.001), shape=(-1, self.configuration.sequence_length, 1))

                # normalized_annotations: [batch_size * document_size, s] 每条评论在每个topic的一个表示,
                self.normalized_annotations = tf.reduce_sum(input_tensor=tf.multiply(x=self.attention_distributions, y=self.word_annotations), axis=1, keep_dims=False)

                # 新增加 [batch_size, document_size, 1]
                self.new_feature = tf.reshape(tensor=tf.matmul(a=self.normalized_annotations, b=self.projection_weights[i]), shape=[-1, self.configuration.document_size, 1])
                self.predicted_featuresu.append(self.new_feature)

                # normalized_annotations: [batch_size, s] 直接求每条评论记录的平均值 来表示用户
                # self.normalized_annotations = tf.reduce_mean(input_tensor=tf.reshape(tensor=self.normalized_annotations, shape=(-1, self.configuration.document_size, self.configuration.s)), axis=1, keep_dims=False)

            # [batch_size, K] concat axis 标识改变某一维的维度 k , batch_size, 1
            # self.predicted_features = tf.concat(values=self.predicted_features, axis=1)
            # [batch_size, document_size, K]
            self.predicted_featuresu = tf.concat(values=self.predicted_featuresu, axis=-1)






class NMRPModel:
    def __init__(self, path_nums, timestamps, f_feature_size, m, random_seed, user_vocabulary_size, item_vocabulary_size, user_document_size,
                 user_review_length, item_document_size, item_review_length, s, dim, sess,
                 unit_type="gru", h=8,
                 K=5, dropout=0.5, l2_reg=0.01, learning_rate=0.001, use_deep=True,
                 deep_layers=[32, 32], deep_layers_activation=tf.nn.relu, dropout_keep_afm=[0.5, 0.5],
                 dropout_keep_deep=[0.5, 0.5], h1=18):
        # 路径个数 默认填充为5
        self.path_nums=path_nums
        # 每种路径节点个数, 默认为4
        self.timestamps=timestamps

        self.f_feature_size = f_feature_size
        # 每个用户每一维的向量表示
        self.h = h

        # embedding_size 嵌入维度
        self.m = m

        self.random_seed = random_seed

        self.deep_layers = deep_layers

        self.l2_reg = l2_reg

        self.use_deep = use_deep

        self.learning_rate = learning_rate

        self.sess = sess

        self.deep_layers_activation = deep_layers_activation

        self.dropout_keep_afm = dropout_keep_afm

        self.dropout_keep_deep = dropout_keep_deep

        self.h1 = h1

        self.vocabulary_size = user_vocabulary_size if user_vocabulary_size > item_vocabulary_size \
            else item_vocabulary_size

        user_configure = AttentionNetworkConfiguration(user_vocabulary_size, m, user_document_size,
                                                       user_review_length, unit_type, s,
                                                       dim, K, dropout)

        item_configure = AttentionNetworkConfiguration(item_vocabulary_size, m, item_document_size,
                                                       item_review_length,
                                                       unit_type, s, dim, K, dropout)

        self.use_attentation = AttentionNetwork("user", user_configure, True)

        self.item_attentation = AttentionNetwork("item", item_configure, True)
        # Review-level Dynamic Topic Co-Attention Layer
        with tf.name_scope("Review-level Dynamic Topic Co-Attention Layer"):

            self.affinityMatrixMedium = tf.get_variable(name="affinityMatrix_weights",
                                                        shape=(self.use_attentation.configuration.K,
                                                               self.item_attentation.configuration.K),
                                                        dtype=tf.float32,
                                                        initializer=tf.random_uniform_initializer(minval=-0.01,
                                                                                                  maxval=0.01))

            self.Wp = tf.Variable(tf.random_uniform([self.use_attentation.configuration.K, self.h1]))

            self.Wq = tf.Variable(tf.random_uniform([self.use_attentation.configuration.K, self.h1]))

            self.Vp = tf.Variable(tf.random_uniform([self.h1, 1]))

            self.Vq = tf.Variable(tf.random_uniform([self.h1, 1]))

            # [batch_size, K, i_document_size]
            self.h_drop_i_re = tf.transpose(self.item_attentation.predicted_featuresu, perm=[0, 2, 1])
            # [batch_size, u_document_size, i_document_size]
            self.affinityMatrixResult = tf.matmul(tf.reshape(tf.matmul(tf.reshape(self.use_attentation.predicted_featuresu,
                                                                                  shape=[-1, self.use_attentation.configuration.K]), self.affinityMatrixMedium),
                                                             shape=[-1, self.use_attentation.configuration.document_size, self.use_attentation.configuration.K]), self.h_drop_i_re)

            self.Hu = tf.nn.tanh(tf.matmul(tf.reshape(self.use_attentation.predicted_featuresu, shape=[-1, self.use_attentation.configuration.K]), self.Wp) +
                               tf.reshape(tf.matmul(self.affinityMatrixResult, tf.reshape(tf.matmul(tf.reshape(self.item_attentation.predicted_featuresu, shape=[-1, self.item_attentation.configuration.K]), self.Wq), shape=[-1, user_document_size, self.h1])), shape=[-1, self.h1]))

            self.Hi = tf.nn.tanh(tf.matmul(tf.reshape(self.item_attentation.predicted_featuresu, shape=[-1, self.item_attentation.configuration.K]), self.Wq)+ tf.reshape(tf.matmul(self.affinityMatrixResult, tf.reshape(tf.matmul(tf.reshape(self.use_attentation.predicted_featuresu, shape=[-1, self.h1]), self.Wp), shape=[-1, item_document_size, self.h1])), shape=[-1, self.h1]))

            self.au = tf.nn.softmax(tf.reshape(tf.matmul(self.Hu, self.Vp), shape=[-1, user_document_size]))

            self.av = tf.nn.softmax(tf.reshape(tf.matmul(self.Hi, self.Vq), shape=[-1, item_document_size]))

            # [None, K]
            self.u = tf.reshape(tf.matmul(tf.expand_dims(self.au, 1), self.use_attentation.predicted_featuresu), shape=[-1, self.use_attentation.configuration.K])

            self.v = tf.reshape(tf.matmul(tf.expand_dims(self.av, 1), self.item_attentation.predicted_featuresu), shape=[-1, self.item_attentation.configuration.K])

        # 异质信息网络
        self.HINMetaAttentionNetwork()

        self.x = tf.concat([self.u, self.v, self.meta_path_embedding])




        self.rating_prediction_module()


    def HINMetaAttentionNetwork(self, filter_sizes=[3], num_filters=4):

        self.ubcatb_input = tf.placeholder(tf.float32, shape=[None, self.path_nums, self.timestamps, self.m],
                                      name='ubcatb_input')

        self.ubub_input = tf.placeholder(tf.float32, shape=[None, self.path_nums, self.timestamps, self.m],
                                    name='ubub_input')

        # [None, self.path_nums, self.timestamps, self.m, 1]
        self.rubcatb_input = tf.expand_dims(self.ubcatb_input, -1)
        self.rubub_input = tf.expand_dims(self.ubub_input, -1)
        # CNN Meta Path Layer
        pooled_outputs_u = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("user_conv-maxpool-%s" % filter_size):
                # Convolution Layer num_filters 100
                filter_shape = [filter_size, self.m, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # batch_size*review_num_u,review_len_u, m, 1
                self.rubcatb_input = tf.reshape(self.rubcatb_input, [-1, self.timestamps, self.m, 1])
                # batch_size*review_num_u ,review_len_u-filter_size+1,1,1,num_fileters
                conv = tf.nn.conv2d(
                    self.rubcatb_input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # len(filter_sizes), batch_size*review_num_u ,1,1,num_filters
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.timestamps - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_u.append(pooled)
        l1 = num_filters * len(filter_sizes)
        self.h_pool_u = tf.concat(pooled_outputs_u, 3)
        # [None, l1]
        self.ubcatb_embedding = tf.reshape(self.h_pool_u, [-1, l1])

        pooled_outputs_i = []

        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("item_conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.m, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                self.rubub_input = tf.reshape(self.rubub_input, [-1, self.timestamps, self.m, 1])
                conv = tf.nn.conv2d(
                    self.rubub_input,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.timestamps - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs_i.append(pooled)
        l1 = num_filters * len(filter_sizes)
        self.h_pool_i = tf.concat(pooled_outputs_i, 3)

        # [None, l1]
        self.ubub_embedding = tf.reshape(self.h_pool_i, [-1, l1])

        # Meta Path Attention Layer

        self.W1= tf.get_variable(name="w1", shape=(1, self.item_attentation.configuration.K*2+l1),
                                                        dtype=tf.float32,
                                                        initializer=tf.random_uniform_initializer(minval=-0.01,
                                                                                                  maxval=0.01))

        self.b1 = tf.get_variable(name="b", shape=(1,),
                                  dtype=tf.float32,
                                  initializer=tf.random_uniform_initializer(minval=-0.01,
                                                                            maxval=0.01))
        a_u_v_p1=tf.nn.relu(tf.matmul(self.W1, tf.concat([self.u, self.v, self.ubcatb_embedding], axis=-1))+self.b1)

        a_u_v_p2=tf.nn.relu(tf.matmul(self.W1, tf.concat([self.u, self.v, self.ubub_embedding], axis=-1))+self.b1)

        a_u_v_p=tf.nn.softmax(tf.concat([a_u_v_p1, a_u_v_p2], axis=-1))

        self.meta_path_embedding=tf.reshape(tf.matmul(tf.transpose(tf.reshape(a_u_v_p,shape=[-1,2,1]),[0,2,1]), tf.concat([self.ubcatb_embedding, self.ubub_embedding], axis=1)), shape=[-1, l1])
        # 我们尝试使用平均注意力机制
        # self.meta_path_embedding=tf.reduce_mean(tf.concat([self.ubcatb_embedding, self.ubub_embedding], axis=-1), axis=-1)


    def rating_prediction_module(self):

        tf.set_random_seed(self.random_seed)

        self.feat_index = tf.placeholder(tf.int32, shape=[None, None],
                                         name="feat_index")  # None * F

        # self.feat_value = tf.concat([self.use_attentation.predicted_features,
        #                              self.item_attentation.predicted_features], axis=-1)
        self.feat_value = self.x

        # 具体的分数信息
        self.label = tf.placeholder(tf.float32, shape=[None], name="label")  # None * 1
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        self.weights = self._initialize_weights()
        # model
        self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],
                                                 self.feat_index)  # None * F * K
        # 维度调整 None * F * 1
        feat_value = tf.reshape(self.feat_value, shape=[-1, self.f_feature_size, 1])

        self.embeddings = tf.multiply(self.embeddings, feat_value)

        # ---------- first order term     表示 wi*xi累加和----------
        self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index)  # None * F * 1
        # reduce_sum 消除其中一个维度
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
        # 防止过拟合 随机丢弃一部分神经元
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0])  # None * F

        # ---------- second order term    转换之后的表示二阶交互特征---------------
        # sum_square part
        self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # square_sum part
        self.squared_features_emb = tf.square(self.embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # second order
        self.y_second_order = 0.5 * tf.subtract(self.summed_features_emb_square,
                                                self.squared_sum_features_emb)  # None * K
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

        # ---------- Deep component    直接连接每一个向量 e1 e2 e3----------
        # self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.latent_factor_size])  # None * (F*K)
        # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
        #
        # for i in range(0, len(self.deep_layers)):
        #     self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]),
        #                          self.weights["bias_%d" % i])  # None * layer[i] * 1
        #     self.y_deep = self.deep_layers_activation(self.y_deep)
        #     self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1 + i])  # dropout at each Deep layer


        concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)

        self.out = tf.reshape(tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"]), shape=[-1])

        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.label, self.out)))
        self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.label, self.out)))
        self.mse = tf.reduce_mean(tf.square(tf.subtract(self.label, self.out)))
        self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.label, self.out))))
        self.accuracy = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.label, self.out))))
        # l2 regularization on weights
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_reg)(self.weights["concat_projection"])


        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                epsilon=1e-8).minimize(self.loss)

    # 初始化权重数据
    def _initialize_weights(self):
        weights = dict()

        # embeddings 嵌入的权重  特征的维度*嵌入的维度
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.f_feature_size, self.h], 0.0, 0.01),
            name="feature_embeddings")  # feature_size * K


        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.f_feature_size, 1], 0.0, 1.0), name="feature_bias")  # feature_size * 1

        # deep layers  隐藏层个数
        num_layer = len(self.deep_layers)
        # 输入层大小  每个字段映射为m维度,然后每个字段映射之后的向量直接连接　　
        # input_size = self.f_feature_size * self.latent_factor_size
        # glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        # weights["layer_0"] = tf.Variable(
        #     np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        # weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
        #                                 dtype=np.float32)  # 1 * layers[0]
        # for i in range(1, num_layer):
        #     glorot = np.sqrt(2.0 / (self.deep_layers[i - 1] + self.deep_layers[i]))
        #     weights["layer_%d" % i] = tf.Variable(
        #         np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i - 1], self.deep_layers[i])),
        #         dtype=np.float32)  # layers[i-1] * layers[i]
        #     weights["bias_%d" % i] = tf.Variable(
        #         np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
        #         dtype=np.float32)  # 1 * layer[i]

        input_size = self.f_feature_size + self.h
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
            dtype=np.float32)  # layers[i-1]*layers[i]
        # 最后结果综合的偏差
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights































