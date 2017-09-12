import numpy as np

from ctypes import cdll
from ctypes import c_double

# calibration constants for X, Y, Z components
ACC_OFFSET_1 = [-202.0281770191, -81.7801589269, 419.6444280899]
GYR_OFFSET_1 = [-15.353925709, -13.0307386379, 13.6007070723]

ACC_SCALE_1 = [0.0023891523, 0.0023878861, 0.0023579114]
GYR_SCALE_1 = 0.001064225

ACC_OFFSET_2 = [122.5744738301, 30.4468158626, 34.4573509276]
GYR_OFFSET_2 = [-11.4545599896, 15.2462993345, 4.4647835565]

ACC_SCALE_2 = [0.0023981123, 0.0023906275, 0.0023649804]
GYR_SCALE_2 = 0.001064225


class Estimator(object):

    def __init__(self, flag=1):
        # load dynamic library
        self.lib = cdll.LoadLibrary(r'D:\tenseinput\data_collection\fusion\libest.dll')


        # set ctypes type requirements
        self.lib.init_est()
        self.lib.update_est.argtypes = [c_double, c_double, c_double, c_double, c_double, c_double, c_double]
        self.lib.q0.restype = c_double
        self.lib.q1.restype = c_double
        self.lib.q2.restype = c_double
        self.lib.q3.restype = c_double
        if flag == 1:
            self.acc_offset = ACC_OFFSET_1
            self.gyr_offset = GYR_OFFSET_1
            self.acc_scale = ACC_SCALE_1
            self.gyr_scale = GYR_SCALE_1
        elif flag == 2:
            self.acc_offset = ACC_OFFSET_2
            self.gyr_offset = GYR_OFFSET_2
            self.acc_scale = ACC_SCALE_2
            self.gyr_scale = GYR_SCALE_2
        else:
            self.acc_offset = ACC_OFFSET_2
            self.gyr_offset = GYR_OFFSET_2
            self.acc_scale = ACC_SCALE_2
            self.gyr_scale = GYR_SCALE_2



    def feed_data(self, dt, raw_gyr, raw_acc):

        # calibrate and convert to SI units
        acc = [0.0, 0.0, 0.0]
        gyr = [0.0, 0.0, 0.0]
        for i in range(3):
            acc[i] = float(raw_acc[i]) + self.acc_offset[i]
            acc[i] *= self.acc_scale[i]
            gyr[i] = float(raw_gyr[i]) + self.gyr_offset[i]
            gyr[i] *= self.gyr_scale

        # attitude estimation
        self.lib.update_est(dt, gyr[0], gyr[1], gyr[2], acc[0], acc[1], acc[2])
        
        q = [0, 0, 0, 0]

        q[0] = self.lib.q0()
        q[1] = self.lib.q1()
        q[2] = self.lib.q2()
        q[3] = self.lib.q3()

        # rotate unit vector by quarternion
        v = np.array(acc)
        u = np.array(q[1:])
        s = q[0]
        vp = 2.0 * np.dot(u,v) * u + (s**2 - np.dot(u,u)) * v + 2.0 * s * np.cross(u,v) - np.array([0,0,9.8])

        return vp