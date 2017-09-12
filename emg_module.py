import threading
import win32file
import time

class Emg_module(threading.Thread):
    def __init__(self, feed_data_func):
        threading.Thread.__init__(self)
        self.feed_data = feed_data_func
        self.fileHandle = win32file.CreateFile("\\\\.\\pipe\\emg_pipe", win32file.GENERIC_READ | win32file.GENERIC_WRITE, 0,
                                          None, win32file.OPEN_EXISTING, 0, None)

    def run(self):
        while True:
            ret, s_bytes = win32file.ReadFile(self.fileHandle, 10240)
            s = s_bytes.decode("utf-8")
            if ret != 0:
                print("Error reading from pipe")
                exit(1)
            if len(s) == 0:
                time.sleep(0.01)
                continue

            for line in s.strip().split('\n'):
                if line.startswith('\0'):
                    line = line[1:]
                row = line.strip().split(' ')
                rowdata = []
                for i, item in enumerate(row):
                    if i == 3 or i == 4:
                        continue
                    rowdata.append(int(item))
                self.feed_data(rowdata)
