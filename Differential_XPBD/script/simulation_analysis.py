import matplotlib.pyplot as plt
import math
import numpy as np
from numpy.lib.format import dtype_to_descr
from pyquaternion import Quaternion
import random
import sympy as sp
import copy
import pdb

from functools import wraps
import time
import copy
from scipy.spatial.transform import Rotation as R
import torch
from torch.autograd import Variable
from torch import optim

from mpl_toolkits.mplot3d import Axes3D
import time
import imageio
import os
from mpl_toolkits.mplot3d import Axes3D
import sys
from matplotlib import cm


class AnalysisTrajDeviation():
    def TrajDeviation(self, waypoints_file, traj_shape):

        #--------- load the data for real traj for the target points --------
        real_traj_target = np.load(waypoints_file + "/" + traj_shape + ".npy")
        #pdb.set_trace()

        #--------- load the data for sim traj for the target points --------\
        files_list = os.listdir(waypoints_file)
        sim_traj_target = []
        sim_traj_control = []
        for index_file in range(len(files_list) - 1):
            sim_nodes = np.load(waypoints_file + "/" +
                                "frame_%d.npy" % index_file)
            sim_traj_target.append(sim_nodes[15, :3])
            sim_traj_control.append(sim_nodes[17, :3])

        #--------- Calculate the real to sim loss and draw --------
        real_to_sim_loss = np.linalg.norm(real_traj_target - sim_traj_target,
                                          axis=1)
        plt.plot(np.arange(len(files_list) - 1),
                 real_to_sim_loss,
                 label="real2sim loss")
        #pdb.set_trace()

        #------ The movement distance for each time -------
        dist = np.linalg.norm(real_traj_target[:-1] - real_traj_target[1:],
                              axis=1)
        dist = np.insert(
            dist, 0, np.linalg.norm(sim_traj_target[0] - np.array([0, 28, 0])))
        dist_percent = np.divide(real_to_sim_loss, dist)

        #plt.plot(np.arange(len(files_list) - 1),
        #         dist,
        #         label="Movement Distance")
        plt.plot(np.arange(len(files_list) - 1),
                 dist_percent,
                 label="Movement Distance Loss Percent")
        plt.legend()
        plt.show()
        pdb.set_trace()
