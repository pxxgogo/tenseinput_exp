import json
import os

import numpy as np
import tensorflow as tf

from mix_model import Model

EMG_IN_CHANNELS = 6
EMG_IN_SIZE = 25
ACC_IN_CHANNELS = 3
ACC_IN_SIZE = 500
INTERVAL_TIME = 5
TENSE_THRESHOLD = -2
RELAX_THRESHOLD = 2


def fft(window_data):
    x_freq = abs(np.fft.rfft(np.array(window_data[0], dtype=np.float32)))
    y_freq = abs(np.fft.rfft(np.array(window_data[1], dtype=np.float32)))
    z_freq = abs(np.fft.rfft(np.array(window_data[2], dtype=np.float32)))
    freqs = [x_freq, y_freq, z_freq]
    return freqs


class Data_collection(object):
    def __init__(self, model_dir, queue_lock, work_queue):
        self.queue_lock = queue_lock
        self.work_queue = work_queue
        config_dir = os.path.join(model_dir, "model_config.json")
        self.config = json.load(open(config_dir))
        self.config["batch_size"] = 1
        session_config = tf.ConfigProto(log_device_placement=False)
        session_config.gpu_options.allow_growth = False
        self.session = tf.Session(config=session_config)
        with tf.variable_scope("model", reuse=None):
            self.model = Model(config=self.config)
        self.session.run(tf.global_variables_initializer())
        new_saver = tf.train.Saver()
        new_saver.restore(self.session, tf.train.latest_checkpoint(
            model_dir))
        self.sample_index = 0
        self.emg_data_index = 0
        self.emg_data = np.zeros([EMG_IN_CHANNELS, EMG_IN_SIZE], dtype=np.float32)
        self.acc_data_index = 0
        self.acc_data = np.zeros([ACC_IN_CHANNELS, ACC_IN_SIZE], dtype=np.float32)
        self.ret_No = 0
        self.current_state = 0

    def draw_ret(self, ret, logits):
        self.queue_lock.acquire()
        self.work_queue.put(ret)
        self.queue_lock.release()
        pass

    def predict(self):
        acc_data = np.concatenate([self.acc_data[:, self.acc_data_index:], self.acc_data[:, :self.acc_data_index]],
                                  axis=1)
        emg_data = np.concatenate([self.emg_data[:, self.emg_data_index:], self.emg_data[:, :self.emg_data_index]],
                                  axis=1)
        # print(acc_data[0])
        acc_fft_data = np.array([fft(acc_data)])
        emg_data = np.array([emg_data])
        fetches = [self.model.predict_op, self.model.logits]
        feed_dict = {}
        feed_dict[self.model.acc_input_data] = acc_fft_data
        feed_dict[self.model.emg_input_data] = emg_data
        predict_val, logits = self.session.run(fetches, feed_dict)
        diff = logits[0][0] - logits[0][1]
        if self.current_state == 0 and diff < TENSE_THRESHOLD:
            self.current_state = 1
        elif self.current_state == 1 and diff > RELAX_THRESHOLD:
            self.current_state = 0
        print(self.ret_No, self.current_state, logits, logits[0][0] - logits[0][1])
        self.ret_No += 1
        self.draw_ret(self.current_state, logits)

    def feed_emg_data(self, data):
        self.emg_data[:, self.emg_data_index] = np.array(data)
        self.emg_data_index += 1
        self.sample_index += 1
        if self.emg_data_index >= EMG_IN_SIZE:
            self.emg_data_index = 0
        if self.sample_index == INTERVAL_TIME:
            self.predict()
            self.sample_index = 0

    def feed_acc_data(self, data):
        self.acc_data[:, self.acc_data_index] = np.array(data)
        self.acc_data_index += 1
        if self.acc_data_index >= ACC_IN_SIZE:
            self.acc_data_index = 0
