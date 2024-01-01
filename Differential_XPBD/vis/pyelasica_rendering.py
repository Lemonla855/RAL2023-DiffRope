#import script.PBDRope
import sys

sys.path.append("..")
from script import PBDRope_dynamics
from script import Visulization
from script import PBDRope
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

# from mpl_toolkits.mplot3d import Axes3D
# import time

if __name__ == '__main__':
    
    ## ================================================================================
    #    Method for Calculating the loss
    ## ================================================================================
    cal_loss_method = "projection"
    cal_loss_target = "centerline_3d_gravity"

    ## ================================================================================
    #    Frame information to modify
    ## ================================================================================
    start_frame = 50  #
    end_frame = 66
    save_dir = "../DATA/save_loss_withConstConvergency/" + cal_loss_method + "/" + cal_loss_target
    use_2D_centerline = False  #To judge whether we need to load the 2D centerline Data

    ## ================================================================================
    #    Data for backward
    ## ================================================================================
    LR_RATE = 0.001  #The learning rate for the optimization(15 to 48:0.005,49 to :0.0005,50:0.01) #For 51 to 58, learning rate is 0.001
    LAYERS = 12  #The number of steps to update other constraint in a step
    STEPS = 30  #The steps  to update the gravity?
    ITERATION = 50  #The nmber of interation to optimize
    backward = True  #Choose to do the forward or backward process

    ## ================================================================================
    #    Data for Visulization
    ## ===========================================================
    visulization_loss_type = cal_loss_method + "/" + cal_loss_target
    index_draw_pclpbdcenterline = [
        30, 28, 43
    ]  # The index must between 16 to 65, this index is for pcl and PBD and Simulation result
    draw_pcl = False  # Whether draw downsampled PCL on the image
    savefig_PCLPBDCenterline_path = cal_loss_method + "/" + cal_loss_target  #Path to save the PCLPBDCenterline plot
    # -------- Information for RealtoSimFigure --------
    RealtoSimFigure_name = cal_loss_method + "_" + cal_loss_target

    ## ================================================================================
    #    Data for PyElasticaRendering
    ## ================================================================================
    PyElasticaRendering_DATA_PATH = save_dir+"/"  # The path to load the original Simulation data
    PyElasticaRendering_OUTPUT_VIDEO_Path = "../DATA/PyElastica/PyElasticaRenderingVideo/" + cal_loss_method + "_" + cal_loss_target + "_multirod"  #The path to output the video
    PyElasticaRendering_OUTPUT_VIDEO_Name = cal_loss_method + "_" + cal_loss_target  #The path to output the video
    PyElasticaRendering_OUTPUT_IMAGES_DIR = "../DATA/PyElastica/PyElasticaRenderingImage/" + cal_loss_method + "_" + cal_loss_target + "_multirod" #The path to ouput the image

    ## ================================================================================
    #   Execute the PyElastica Rendering
    ## ================================================================================
    PyElasticaRendering = Visulization.PyElasticRenderingMultiRod()
    PyElasticaRendering.ExecuteRendering(
        PyElasticaRendering_DATA_PATH, PyElasticaRendering_OUTPUT_VIDEO_Path,
        PyElasticaRendering_OUTPUT_VIDEO_Name,
        PyElasticaRendering_OUTPUT_IMAGES_DIR)
