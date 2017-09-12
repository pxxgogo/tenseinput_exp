import argparse
import queue
import socket
import threading
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from acc_module import Acc_module
from data_collection import Data_collection
from emg_module import Emg_module

queue_lock = threading.Lock()
work_queue = queue.Queue(10)
fig = plt.figure()
fig.patches.append(mpatches.Circle([0.5, 0.5], 0.1, transform=fig.transFigure))
result_pointer = 0
result_window = np.zeros(1)
addr = ('192.168.1.154', 12346)
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


def update_result_window(ret):
    global result_pointer
    global result_window
    result_window[result_pointer] = ret
    ret_raw = np.mean(result_window)
    if ret_raw > 0:
        ret_raw = 1
    result_pointer += 1
    if result_pointer == result_window.shape[0]:
        result_pointer = 0
    return ret_raw


def draw_ret(ret):
    if not ret == -1:
        ret = update_result_window(ret)
        if ret == 0:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
        else:
            fig.patches.append(mpatches.Circle([0.5, 0.5], 0.25, transform=fig.transFigure, color=(ret, 0, 0)))
        fig.show()
        plt.pause(0.05)
        fig.clf()


def send(ret):
    s.sendto((str(time.time()) + " " + str(ret)).encode(), addr)


def process_data():
    while True:
        queue_lock.acquire()
        if not work_queue.empty():
            data = work_queue.get()
            queue_lock.release()
            send(data)
            # draw_ret(data)
        else:
            queue_lock.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=r"..\tenseinput_model\mixed\model_single_window_3")
    args = parser.parse_args()
    model_dir = args.model_dir
    data_collection = Data_collection(model_dir, queue_lock, work_queue)
    emg_module = Emg_module(data_collection.feed_emg_data)
    acc_module = Acc_module(data_collection.feed_acc_data)
    emg_module.start()
    acc_module.start()
    process_data()
