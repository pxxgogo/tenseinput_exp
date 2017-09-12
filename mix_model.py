import tensorflow as tf

flags = tf.flags
logging = tf.logging

flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")

FLAGS = flags.FLAGS


def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class Model(object):
    """The RNN model."""

    def __init__(self, config):
        self.batch_size = config['batch_size']
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.input_channel = config['input_channel']
        self.acc_model_structure = config['acc_model_structure']
        self.emg_model_structure = config['emg_model_structure']
        self.acc_regularized_lambda = config.get("acc_regularized_lambda", 0.5)
        self.emg_regularized_lambda = config.get("emg_regularized_lambda", 0.5)
        self.regularized_flag = config.get("regularized_flag", True)
        self.multi_window_flag = config.get("multi_window_flag", 0)
        self.acc_sequence_flag = config.get("acc_sequence_flag", True)
        self.emg_sequence_flag = config.get("emg_sequence_flag", True)
        self.sequence_size = config.get("sequence_size", [1, 1])
        self.model_type = config.get('model_type', 0)
        acc_data = self.acc_process()
        emg_data = self.emg_process()

        self._targets = tf.placeholder(tf.int16, [self.batch_size, self.output_size])
        if self.model_type == 0:
            data = tf.concat([acc_data, emg_data], axis=1)
        elif self.model_type == 1:
            data = acc_data
        else:
            data = emg_data
        softmax_w = tf.get_variable(
            "softmax_w", [data.shape[1], self.output_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [self.output_size], dtype=data_type())
        self.logits = logits = tf.matmul(data, softmax_w) + softmax_b
        # print(self.logits.shape)
        self.costs = tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=logits)
        loss = tf.reduce_mean(self.costs)
        self._cost = cost = loss
        self._predict_op = tf.argmax(logits, 1)

    def acc_process(self):
        if self.acc_sequence_flag and self.multi_window_flag == 1:
            self._acc_input_data = tf.placeholder(data_type(),
                                                  [self.batch_size, self.sequence_size[0], self.input_channel[0],
                                                   self.input_size[0]])
        else:
            self._acc_input_data = tf.placeholder(data_type(),
                                                  [self.batch_size, self.input_channel[0], self.input_size[0]])
        acc_data = self._acc_input_data

        layer_No = 0
        for i, layer in enumerate(self.acc_model_structure):
            net_type = layer["net_type"]
            repeated_times = layer.get("repeated_times", 1)
            while repeated_times > 0:
                if net_type == "LSTM":
                    acc_data = self.add_lstm_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        hidden_size=layer["hidden_size"]
                    )
                elif net_type == "RESNET":
                    acc_data = self.add_resnet_layer(
                        data=acc_data,
                        input=self._acc_input_data)
                elif net_type == "CNN":
                    if len(acc_data.shape) == 3:
                        acc_data = tf.reshape(acc_data, [self.batch_size, self.input_channel[0], -1, 1])
                    acc_data = self.add_conv_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.acc_regularized_lambda
                    )
                    acc_data = self.add_pool_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        pool_size=layer["pool_size"],
                        strides=[1, 1, 1, 1],
                        pool_type=layer["pool_type"]
                    )
                elif net_type == "GCNN":
                    if len(acc_data.shape) == 3:
                        acc_data = tf.reshape(acc_data, [self.batch_size, self.input_channel[0], -1, 1])
                    acc_data = self.add_conv_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.acc_regularized_lambda
                    )
                    acc_data_g = self.add_conv_layer(
                        type='acc_gcnn',
                        No=layer_No,
                        input=acc_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.acc_regularized_lambda
                    )
                    acc_data = tf.multiply(acc_data, tf.sigmoid(acc_data_g))
                    acc_data = self.add_pool_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        pool_size=layer["pool_size"],
                        strides=[1, 1, 1, 1],
                        pool_type=layer["pool_type"]
                    )
                elif net_type == "DENSE":
                    keep_prob = 1.0
                    acc_data = self.add_dense_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        output_size=layer["output_size"],
                        keep_prob=keep_prob,
                        regularized_lambda=self.acc_regularized_lambda
                    )
                elif net_type == "CNN3D":
                    if len(acc_data.shape) < 5:
                        acc_data = tf.reshape(acc_data,
                                              [self.batch_size, self.sequence_size[0], self.input_channel[0], -1, 1])
                    acc_data = self.add_conv3d_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.acc_regularized_lambda
                    )
                    acc_data = self.add_pool3d_layer(
                        type='acc',
                        No=layer_No,
                        input=acc_data,
                        pool_size=layer["pool_size"],
                        strides=[1, 1, 1, 1, 1],
                        pool_type=layer["pool_type"]
                    )
                repeated_times -= 1
                layer_No += 1

        acc_data = tf.reshape(acc_data, [self.batch_size, -1])

        return acc_data

    def emg_process(self):
        if self.emg_sequence_flag and self.multi_window_flag == 1:
            self._emg_input_data = tf.placeholder(data_type(),
                                                  [self.batch_size, self.sequence_size[1], self.input_channel[1],
                                                   self.input_size[1]])
        else:
            self._emg_input_data = tf.placeholder(data_type(),
                                                  [self.batch_size, self.input_channel[1], self.input_size[1]])
        emg_data = self._emg_input_data
        layer_No = 0
        for i, layer in enumerate(self.emg_model_structure):
            net_type = layer["net_type"]
            repeated_times = layer.get("repeated_times", 1)
            while repeated_times > 0:
                if net_type == "LSTM":
                    emg_data = self.add_lstm_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        hidden_size=layer["hidden_size"]
                    )
                elif net_type == "RESNET":
                    emg_data = self.add_resnet_layer(
                        data=emg_data,
                        input=self._emg_input_data)
                elif net_type == "CNN":
                    if len(emg_data.shape) == 3:
                        emg_data = tf.reshape(emg_data, [self.batch_size, self.input_channel[1], -1, 1])
                    emg_data = self.add_conv_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.emg_regularized_lambda
                    )
                    emg_data = self.add_pool_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        pool_size=layer["pool_size"],
                        strides=[1, 1, 1, 1],
                        pool_type=layer["pool_type"]
                    )
                elif net_type == "DENSE":
                    keep_prob = 1.0
                    emg_data = self.add_dense_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        output_size=layer["output_size"],
                        keep_prob=keep_prob,
                        regularized_lambda=self.emg_regularized_lambda
                    )
                elif net_type == "CNN3D":
                    if len(emg_data.shape) < 5:
                        emg_data = tf.reshape(emg_data,
                                              [self.batch_size, self.sequence_size[1], self.input_channel[1], -1, 1])
                    emg_data = self.add_conv3d_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        filter_size=layer["filter_size"],
                        out_channels=layer["out_channels"],
                        filter_type=layer["filter_type"],
                        regularized_lambda=self.emg_regularized_lambda
                    )
                    emg_data = self.add_pool3d_layer(
                        type='emg',
                        No=layer_No,
                        input=emg_data,
                        pool_size=layer["pool_size"],
                        strides=[1, 1, 1, 1, 1],
                        pool_type=layer["pool_type"]
                    )
                repeated_times -= 1
                layer_No += 1
        emg_data = tf.reshape(emg_data, [self.batch_size, -1])
        return emg_data

    def add_lstm_layer(self, type, No, input, hidden_size, keep_prob=1.0):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=0.0, state_is_tuple=True,
                                                 reuse=tf.get_variable_scope().reuse)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        with tf.variable_scope("lstm_layer_%s_%d" % (type, No)):
            outputs, last_states = tf.nn.dynamic_rnn(
                cell=lstm_cell,
                dtype=data_type(),
                inputs=input)
        return tf.convert_to_tensor(outputs)

    def add_conv_layer(self, type, No, input, filter_size, out_channels, filter_type, regularized_lambda,
                       strides=[1, 1, 1, 1], r_flag=True):
        with tf.variable_scope("conv_layer_%s_%d" % (type, No)):
            W = tf.get_variable('filter', [filter_size[0], filter_size[1], input.shape[3], out_channels])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv2d(
                input,
                W,
                strides=strides,
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_conv3d_layer(self, type, No, input, filter_size, out_channels, filter_type, regularized_lambda,
                         r_flag=True):
        with tf.variable_scope("conv3d_layer_%s_%d" % (type, No)):
            W = tf.get_variable('filter',
                                [filter_size[0], filter_size[1], filter_size[2], input.shape[4], out_channels])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [out_channels])
            conv = tf.nn.conv3d(
                input,
                W,
                strides=[1, 1, 1, 1, 1],
                padding=filter_type,
                name='conv'
            )
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        return h

    def add_pool_layer(self, type, No, input, pool_size, strides, pool_type):
        for i in range(2):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%s_%d" % (type, No)):
            pooled = tf.nn.max_pool(
                input,
                ksize=[1, pool_size[0], pool_size[1], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def add_pool3d_layer(self, type, No, input, pool_size, strides, pool_type):
        for i in range(3):
            if pool_size[i] == -1:
                pool_size[i] = input.shape[1 + i]
        with tf.variable_scope("pool_layer_%s_%d" % (type, No)):
            pooled = tf.nn.max_pool3d(
                input,
                ksize=[1, pool_size[0], pool_size[1], pool_size[2], 1],
                padding=pool_type,
                strides=strides,
                name='pool'
            )
        return pooled

    def get_length(self, input):
        ret = 1
        for i in range(1, len(input.shape)):
            ret *= int(input.shape[i])
        return ret

    def add_dense_layer(self, type, No, input, output_size, keep_prob, regularized_lambda, r_flag=True):
        with tf.variable_scope("dense_layer_%s_%d" % (type, No)):
            input_length = self.get_length(input)
            W = tf.get_variable('dense', [input_length, output_size])
            if r_flag:
                tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularized_lambda)(W))
            b = tf.get_variable('bias', [output_size])
            data = tf.reshape(input, [-1, int(input_length)])
            data = tf.nn.relu(tf.matmul(data, W) + b)
            if keep_prob < 1.0:
                data = tf.nn.dropout(data, keep_prob)
        return data

    def add_resnet_layer(self, data, input):
        return data + input

    @property
    def acc_input_data(self):
        return self._acc_input_data

    @property
    def emg_input_data(self):
        return self._emg_input_data

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def predict_op(self):
        return self._predict_op
