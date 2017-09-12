import threading
import time
import serial
from estimator import Estimator


def read_data(data, offset):
    arr = []
    for i in range(3):
        arr.append(int(data[i * 2 + offset] << 8 | data[i * 2 + 1 + offset]))
        if arr[i] >= 2 ** 15:
            arr[i] -= 2 ** 16
    return arr


class Acc_module(threading.Thread):
    def __init__(self, feed_data_func):
        threading.Thread.__init__(self)
        self.feed_data = feed_data_func
        self.serial_port = serial.Serial('COM3', 115200)
        self.est = Estimator()

    def run(self):
        lastTp = time.time()
        while True:
            data = self.serial_port.read(14)
            raw_acc = read_data(data, 0)
            raw_gyr = read_data(data, 8)

            tp = time.time()
            dt = tp - lastTp
            lastTp = tp

            # attitude estimation
            vp = self.est.feed_data(dt, raw_gyr, raw_acc)
            self.feed_data(vp)
