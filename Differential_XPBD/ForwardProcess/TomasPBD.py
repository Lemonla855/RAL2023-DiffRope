#import script.PBDRope
import sys

sys.path.append("..")
from script import PBDRope_dynamics
from script import Visulization
from script import PBDRope
from script import ForwardProcess
#from script import pbd_rope_render
# import matplotlib.pyplot as plt
# import math
# import numpy as np
# from numpy.lib.format import dtype_to_descr
# from pyquaternion import Quaternion
# import random
# import sympy as sp
# import copy
# import pdb

# from functools import wraps
# import time
# import copy
# from scipy.spatial.transform import Rotation as R
# import torch
# from torch.autograd import Variable
# from torch import optim

# from mpl_toolkits.mplot3d import Axes3D# Solver = PBDRope.ExecuteBackward()
# Solver.Execute(start_frame, end_frame, save_dir, LR_RATE, LAYERS, STEPS,
#                ITERATION, use_2D_centerline, backward, cal_loss_method,
#                cal_loss_target)

# import time

if __name__ == '__main__':

    ## ================================================================================
    #    Method for Calculating the loss
    ## ================================================================================
    cal_loss_method = "projection"
    cal_loss_target = "centerline_3d_gravity"
    dynamics = False

    Tomas = True

    ## ================================================================================
    #    Frame information to modify
    ## ================================================================================
    start_frame = 30  #
    end_frame = 31
    save_dir = "../DATA/Forward/" + "TomasPBD_30"
    use_2D_centerline = False  #To judge whether we need to load the 2D centerline Data

    ## ================================================================================
    #    Data for backward
    ## ================================================================================
    LR_RATE = 0.001  #The learning rate for the optimization(15 to 48:0.005,49 to :0.0005,50:0.01) #For 51 to 58, learning rate is 0.001
    LAYERS = 12  #The number of steps to update other constraint in a step
    STEPS = 30  #The steps  to update the gravity?
    ITERATION = 50  #The nmber of interation to optimize
    backward = False  #Choose to do the forward or backward process

    # PyElasticaRendering_OUTPUT_IMAGES_DIR = "../DATA/PyElastica/PyElasticaRenderingImage/" + cal_loss_method + "_" + cal_loss_target  #The path to ouput the image

    # # ================================================================================
    # #  Execute the solve
    # # ================================================================================
    Solver = ForwardProcess.ExecuteBackward()
    Solver.Execute(start_frame, end_frame, save_dir, LR_RATE, LAYERS, STEPS,
                   ITERATION, use_2D_centerline, backward, cal_loss_method,
                   cal_loss_target, dynamics, Tomas)
