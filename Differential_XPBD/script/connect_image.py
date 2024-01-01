from posix import listdir
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
import os
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches
from scipy.interpolate import splrep, splev
from matplotlib.patches import Circle
from sympy import Plane, Point, Point3D
import cv2 as cv
# class ConstraintSatisfication():
#     def __init__(self):
#         self.node_num = 20
#         self.setGoalState()

#     def AnalysisC_bending(self, sim_node):

#         _, _, curr_quaternion = self.Quaternion(sim_node)

#         # q0[0] = torch.as_tensor([0.682700, -0.318297, -0.000019, 0.657730])
#         # q1[0] = torch.as_tensor([0.804539, -0.374640, -0.000135, 0.460827])
#         P = torch.zeros_like(curr_quaternion[:-1])
#         P[:, 3] = curr_quaternion[:-1, 3]
#         P[:, :3] = -1 * (curr_quaternion[:-1, :3])  #conjugate
#         Q = curr_quaternion[1:, :]

#         # Qa = torch.as_tensor(
#         #    [-0.018268, 0.217283, -0.182538, 0.958715]).reshape(-1, 4)
#         # Qb = torch.as_tensor(
#         #    [0.365269, -0.782895, -0.027047, 0.502914]).reshape(-1, 4)

#         omega_PQ, omega_pq0 = self.mul_quaternion(P, Q)
#         omega = torch.hstack((omega_PQ, omega_pq0.reshape(-1, 1)))
#         #omega = torch.div(omega, torch.linalg.norm(omega, dim=1))

#         # self.darbouxRest[0] = torch.as_tensor(
#         #    [0.214160, -0.099520, -0.000316, 0.971716])

#         delta_omega = torch.zeros_like(omega)

#         #darboux_method_A = self.getDarbouxVect(quat)
#         #pdb.set_trace()

#         omega_plus = omega + self.DarbouxVector
#         omega_minus = omega - self.DarbouxVector
#         for i in range(omega_minus.shape[0]):
#             if torch.linalg.norm(omega_minus[i], ord=2) > torch.linalg.norm(
#                     omega_plus[i], ord=2):
#                 delta_omega[i] = omega_plus[i]
#             else:
#                 delta_omega[i] = omega_minus[i]

#         C_bending = torch.as_tensor(delta_omega[:, :3])
#         return C_bending

#     def AnalysisC_strain(self, sim_node):
#         pos_a = sim_node[:-1, :3]
#         pos_b = sim_node[1:, :3]
#         _, _, curr_quaternin = self.Quaternion(sim_node)
#         #pdb.set_trace()
#         d3 = torch.zeros((sim_node.shape[0] - 1, 3))
#         #pdb.set_trace()
#         d3[:,
#            0] = 2.0 * (torch.add(curr_quaternin[:, 0] * curr_quaternin[:, 2],
#                                  curr_quaternin[:, 3] * curr_quaternin[:, 1]))

#         d3[:, 1] = 2.0 * torch.sub(curr_quaternin[:, 1] * curr_quaternin[:, 2],
#                                    curr_quaternin[:, 3] * curr_quaternin[:, 0])

#         d3[:, 2] = curr_quaternin[:, 3] * curr_quaternin[:, 3] - curr_quaternin[:, 0] * \
#             curr_quaternin[:, 0] - curr_quaternin[:, 1] * curr_quaternin[:, 1] + curr_quaternin[:, 2] * curr_quaternin[:, 2]
#         #pdb.set_trace()
#         C_strain = torch.div(pos_b - pos_a, self.goal_dist.reshape(-1, 1)) - d3

#         return C_strain

#     def AnalysisC_dist(self, sim_node):
#         diff_pos = torch.diff(sim_node, dim=0).float()
#         diff_pos_norm = torch.linalg.norm(diff_pos, dim=1)

#         C_dist = diff_pos_norm - self.goal_dist[:]
#         return C_dist

#     def setGoalState(self):

#         # define goal position of particles
#         # goal_pos = torch.stack(
#         #    [torch.arange(self.node_num), 2 * torch.arange(self.node_num), 3 * torch.arange(self.node_num)])

#         # self.node_pos_goal = torch.transpose(goal_pos, 0, 1)

#         # ------- Initialize Goal Position for the node -------
#         self.node_pos_goal = torch.zeros((self.node_num, 3))
#         centerline_3d_gravity = np.load(
#             "../DATA/DATA_BaxterRope/downsampled_pcl/initialization.npy") * 1.0
#         min = centerline_3d_gravity[0]
#         max = centerline_3d_gravity[-1]

#         distance = (max - min) / (self.node_num - 1)
#         for i in range(self.node_num):
#             self.node_pos_goal[i] = torch.as_tensor(min + i * distance) * 1.0

#         # ------- Initialize Goal Radius for the node -------
#         self.node_radius_goal = torch.ones(self.node_num)
#         for i in range((self.node_num)):
#             self.node_radius_goal[i] = 0.01

#         # ------- Initialize Rest Lenght for the node -------
#         self.goal_dist = self.RestLength(self.node_pos_goal)
#         # self.goal_dist[:] = 0.0571
#         # self.goal_dist[:] = 0.0601

#         # ------- Initialize Goal Quaternion for the node -------
#         self.node_rotation, self.node_length_goal, self.node_quaternion_goal = self.Quaternion(
#             self.node_pos_goal)

#         # ------- Initialize DarbouxRest for the node -------
#         self.DarbouxVector = self.DarbouxRest(self.node_pos_goal,
#                                               self.node_quaternion_goal)

#     # ------- Rest Length -------
#     def RestLength(self, pos):
#         diff_pos = torch.diff(pos, dim=0).float()
#         diff_pos = torch.linalg.norm(diff_pos, dim=1)
#         return diff_pos

#     # ------- Quaternion -------
#     def Quaternion(self, pos):
#         """
#         Find the rotation matrix that aligns vec1 to vec2
#         :param vec1: A 3d "source" vector
#         :param vec2: A 3d "destination" vector
#         :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
#         """
#         node_num = self.node_num
#         source_vec = np.array([0, 0, 1])  #Source Vector
#         q = torch.zeros((node_num - 1, 4))  #Quanternion mateix
#         rest_length = torch.ones((node_num - 1, 1))
#         rot = torch.zeros((node_num - 1, 3, 3))  #rotation matrix

#         for i in range(pos.shape[0] - 1):
#             #pdb.set_trace()

#             pos_a, pos_b = pos[i, :3], pos[i + 1, :3]
#             ab_vec = torch.as_tensor(
#                 pos_b -
#                 pos_a) * 1.0  # ab_vec is the vector between pt a and pt b
#             ab_vec = (ab_vec / torch.linalg.norm(torch.FloatTensor(ab_vec)))
#             ab_vec = ab_vec.detach().numpy()
#             source_vec_normalized, ab_vec_normalized = (
#                 source_vec / np.linalg.norm(source_vec)).reshape(3), (
#                     ab_vec / np.linalg.norm(ab_vec)).reshape(3)

#             v = np.cross(source_vec_normalized, ab_vec_normalized)
#             c = np.dot(source_vec_normalized, ab_vec_normalized)
#             s = np.linalg.norm(v)

#             kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
#                              [-v[1], v[0], 0]])
#             rotation_matrix = np.eye(3) + kmat + \
#                 kmat.dot(kmat) * ((1 - c) / (s ** 2))
#             r = R.from_matrix(rotation_matrix)
#             rot[i] = torch.as_tensor(r.as_matrix())
#             qc = r.as_quat()  # x,y,z,w'

#             q[i] = torch.tensor(qc)
#             rest_length[i] = torch.linalg.norm(
#                 torch.as_tensor(pos_b - pos_a) * 1.0)
#         return rot, rest_length, q

#     # ------- DarbouxRest -------
#     def DarbouxRest(self, target, q):
#         darbouxvector = torch.zeros((len(target) - 2, 4))

#         for i in range(len(target) - 2):

#             q0 = q[i, :]
#             q1 = q[i + 1, :]
#             omega = Quaternion(q0[3], q0[0], q0[1], q0[2]).conjugate * \
#                 Quaternion(q1[3], q1[0], q1[1], q1[2])
#             w, x, y, z = omega.elements[0], omega.elements[1], omega.elements[
#                 2], omega.elements[3]
#             darbouxRest = torch.as_tensor([x, y, z, w])
#             darbouxRest = darbouxRest / np.linalg.norm(darbouxRest)
#             # omega_plus = darbouxRest + np.array([0, 0, 0, 1])
#             # omega_minus=darbouxRest - np.array([0, 0, 0, 1])
#             # if np.linalg.norm(omega_minus,ord=2)>np.linalg.norm(omega_plus,ord=2):
#             #		darbouxRest *= -1.0
#             darbouxvector[i] = darbouxRest

#         return darbouxvector

#     def mul_quaternion(self, p, q):
#         # p and q are represented as x,y,z,w
#         p_scalar, q_scalar = p[:, 3], q[:, 3]
#         p_imag, q_imag = p[:, :3], q[:, :3]
#         quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
#         quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
#             -1, 1) * p_imag + torch.cross(p_imag, q_imag)
#         # omega = torch.hstack((quat_imag, quat_scalar.reshape(-1, 1)))

#         return quat_imag, quat_scalar

#     # ------- Draw Constraint plotting for each frame -------
#     def DrawConstraint(self, load_loc, save_loc):
#         if not os.path.exists(save_loc):
#             os.makedirs(save_loc)
#         #pdb.set_trace()
#         files = listdir(load_loc)
#         for file in files:
#             #pdb.set_trace()
#             if os.path.splitext(file)[1] == ".npy":
#                 if int(os.path.splitext(file)[0].split('_')[1]) == 14:
#                     continue
#                 frame_index = int(os.path.splitext(file)[0].split('_')
#                                   [1])  #The index of the frame from npy file
#                 sim_node = torch.as_tensor(
#                     np.load(load_loc +
#                             "/frame_%d.npy" % frame_index))[:, :3].float()

#                 constraint = ConstraintSatisfication()
#                 #pdb.set_trace()
#                 C_bending = constraint.AnalysisC_bending(sim_node)
#                 C_strain = constraint.AnalysisC_strain(sim_node)
#                 C_dist = constraint.AnalysisC_dist(sim_node)
#                 C_bending_plot = torch.linalg.norm(C_bending, dim=1)
#                 C_strain_plot = torch.linalg.norm(C_strain, dim=1)
#                 C_dist_plot = abs(C_dist)

#                 C_bending_plot = np.insert(C_bending_plot, 18, [0, 0])
#                 C_strain_plot = np.insert(C_strain_plot, 19, [0])
#                 C_dist_plot = np.insert(C_dist_plot, 19, [0])
#                 #pdb.set_trace()

#                 plt.plot(np.arange(0, 20, 1).astype(np.str),
#                          C_bending_plot,
#                          label="C_bending")
#                 plt.plot(np.arange(0, 20, 1), C_strain_plot, label="C_strain")
#                 plt.plot(np.arange(0, 20, 1), C_dist_plot, label="C_dist")
#                 plt.legend()
#                 plt.savefig(save_loc + "/frame_%d.png" % frame_index)
#                 plt.cla()

#     # ------- Draw Constraint plotting for all frames -------
#     def DrawConstraintForAllFrames(self, load_loc, save_loc):
#         if not os.path.exists(save_loc):
#             os.makedirs(save_loc)
#         #pdb.set_trace()
#         files = listdir(load_loc)
#         C_bending_plot = []
#         C_strain_plot = []
#         C_dist_plot = []
#         for file in files:
#             # ------- load the data --------
#             if os.path.splitext(file)[1] == ".npy":
#                 if int(os.path.splitext(file)[0].split('_')[1]) == 14:
#                     continue
#                 frame_index = int(os.path.splitext(file)[0].split('_')
#                                   [1])  #The index of the frame from npy file
#                 sim_node = torch.as_tensor(
#                     np.load(load_loc +
#                             "/frame_%d.npy" % frame_index))[:, :3].float()

#                 # ------- Evaluate the constraints exits or not --------
#                 constraint = ConstraintSatisfication()
#                 C_bending = constraint.AnalysisC_bending(sim_node)
#                 C_strain = constraint.AnalysisC_strain(sim_node)
#                 C_dist = constraint.AnalysisC_dist(sim_node)
#                 C_bending_plot.append(
#                     torch.max(torch.linalg.norm(C_bending, dim=1)))
#                 C_strain_plot.append(
#                     torch.max(torch.linalg.norm(C_strain, dim=1)))
#                 C_dist_plot.append(torch.max(torch.linalg.norm(C_dist, dim=0)))

#         plt.plot(np.arange(0, len(C_bending_plot), 1),
#                  C_bending_plot,
#                  label="C_bending")
#         plt.plot(np.arange(0, len(C_bending_plot), 1),
#                  C_strain_plot,
#                  label="C_strain")
#         plt.plot(np.arange(0, len(C_bending_plot), 1),
#                  C_dist_plot,
#                  label="C_dist")
#         plt.legend()
#         plt.savefig(save_loc + "/frames_all.png")
#         plt.cla()


## ================================================================================
#    Parameter Estimation Analysis
## ================================================================================
class ParaEstiAnalysis():
    def PlotParaEstiAnalysis(self, para_file, num_frame):
        C_dist = []
        C_bending = []
        C_strain = []
        w_gravity = []
        w_dis = []
        w_strain = []
        wq_strain = []
        wq_bending = []
        w_SOR = []
        loss = []
        for itr in range(100):
            file_name = para_file + "/frame_" + str(
                num_frame) + "_itr_%d.npy" % itr

            optimized_data = np.load(file_name)
            # --------  Weight Information -------
            w_gravity.append(abs(optimized_data[0, 0]))
            w_dis.append(optimized_data[1, 0])
            w_strain.append(optimized_data[2, 0])
            wq_strain.append(optimized_data[3, 0])
            wq_bending.append(optimized_data[4, 0])
            w_SOR.append(optimized_data[5, 0])
            loss.append(optimized_data[6, 0])
            # --------  Constraint Information -------
            C_dist.append(np.max(optimized_data[:, 9]))
            C_bending.append(np.max(optimized_data[:, 10]))
            C_strain.append(np.max(optimized_data[:, 11]))

        # plt.plot(np.arange(0, 100, 1), w_gravity, label="w_gravity")
        # #plt.plot(np.arange(0, 50, 1), loss, label="loss")
        # plt.plot(np.arange(0, 100, 1), C_dist, label="C_dist")
        # plt.plot(np.arange(0, 100, 1), C_bending, label="C_bending")
        #plt.plot(np.arange(0, 100, 1), C_strain, label="C_strain")
        plt.plot(np.arange(0, 100, 1), loss, label="loss")
        plt.savefig(para_file + "optimal_result_frame_" + str(num_frame) +
                    ".png")
        plt.legend()
        plt.show()

        minimum_loss_index = np.argmin(loss)
        print("minimum_loss_index:", minimum_loss_index)
        print("w_gravity :", w_gravity[minimum_loss_index])
        print("w_dis     :", w_dis[minimum_loss_index])
        print("w_strain  :", w_strain[minimum_loss_index])
        print("wq_strain :", wq_strain[minimum_loss_index])
        print("wq_bending:", wq_bending[minimum_loss_index])
        print("w_SOR     :", w_SOR[minimum_loss_index])
        print("loss      :", loss[minimum_loss_index])


## ================================================================================
#    Simulation Result Analysis
## ================================================================================
class SimNodeResultAnalysis():
    def PlotSimNodeResultAnalysis(self, para_file, save_loc_analysis):
        all_files = listdir(para_file)

        for file in all_files:
            # -------- Initialize the save information --------
            C_dist = []
            C_bending = []
            C_strain = []
            loss = []
            if os.path.splitext(file)[1] == ".npy":
                if int(os.path.splitext(file)[0].split('_')[1]) < 30:
                    continue
                num_frame = int(os.path.splitext(file)[0].split('_')[1])
                sim_data = np.load(para_file + "/" + file)
                # -------- Save Constraint Information --------
                C_dist.append(sim_data[:, 8])
                C_bending.append(sim_data[:, 9])
                C_strain.append(sim_data[:, 10])
                #loss.append(sim_data[:, 11])
                # -------- Plot Constraint Information -------
                #pdb.set_trace()
                plt.plot(np.arange(20).astype(dtype=np.str),
                         np.array(C_dist).reshape(-1),
                         label="C_dist")
                plt.plot(np.arange(18).astype(dtype=np.str),
                         np.array(C_bending)[0, :-2].reshape(-1),
                         label="C_bending")
                # plt.plot(np.arange(19).astype(dtype=np.str),
                #          np.array(C_strain)[0, :-1].reshape(-1),
                #          label="C_strain")
                plt.legend()
                plt.savefig(save_loc_analysis + "/frame_no_strain" +
                            str(num_frame) + ".png")
                plt.cla()
                #plt.show()
                #pdb.set_trace()

    def PlotSimNodeResultAnalysisALLFrames(self, para_file):
        all_files = listdir(para_file)

        C_dist = []
        C_bending = []
        C_strain = []
        loss = []

        for file in all_files:
            # -------- Initialize the save information --------
            if os.path.splitext(file)[1] == ".npy":
                if int(os.path.splitext(file)[0].split('_')[1]) < 30:
                    continue
                num_frame = int(os.path.splitext(file)[0].split('_')[1])
                sim_data = np.load(para_file + "/" + file)
                # -------- Save Constraint Information --------
                C_dist.append(np.max(sim_data[:, 8]))
                C_bending.append(np.max(sim_data[:, 9]))
                C_strain.append(np.max(sim_data[:, 10]))
                loss.append(sim_data[:, 11])
        # -------- Plot Constraint Information -------
        pdb.set_trace()
        plt.plot(np.arange(len(C_dist)),
                 np.array(C_dist).reshape(-1),
                 label="C_dist")
        plt.plot(np.arange(len(C_dist)),
                 np.array(C_bending).reshape(-1),
                 label="C_bending")
        plt.plot(np.arange(len(C_dist)),
                 np.array(C_strain).reshape(-1),
                 label="C_strain")
        plt.legend()
        plt.savefig("./DATA/PlotSimNodeResultAnalysisALLFrames.png")
        plt.cla()
        #plt.show()
        #pdb.set_trace()

    def PlotConstraintForEachNode(self, para_file, save_loc_analysis):
        all_files = listdir(para_file)

        C_dist = []
        C_bending = []
        C_strain = []
        loss = []

        for file in all_files:
            # -------- Initialize the save information --------
            if os.path.splitext(file)[1] == ".npy":
                if int(os.path.splitext(file)[0].split('_')[1]) < 30:
                    continue
                num_frame = int(os.path.splitext(file)[0].split('_')[1])
                sim_data = np.load(para_file + "/" + file)
                # -------- Save Constraint Information --------
                C_dist.append(sim_data[:, 8])
                C_bending.append(sim_data[:, 9])
                C_strain.append(sim_data[:, 10])
                loss.append(sim_data[:, 11])

        C_dist = np.array(C_dist)
        C_bending = np.array(C_bending)
        C_strain = np.array(C_strain)

        C_dist_mean = np.mean(C_dist, axis=0)
        C_bending_mean = np.mean(C_bending, axis=0)
        C_strain_mean = np.mean(C_strain, axis=0)
        #pdb.set_trace()

        # -------- Plot Constraint Information -------
        #pdb.set_trace()
        plt.plot(np.arange(len(C_dist_mean)),
                 np.array(C_dist_mean).reshape(-1),
                 label="C_dist_mean")
        plt.plot(np.arange(len(C_bending_mean)),
                 np.array(C_bending_mean).reshape(-1),
                 label="C_bending_mean")
        # plt.plot(np.arange(len(C_strain_mean)),
        #          np.array(C_strain_mean).reshape(-1),
        #          label="C_strain_mean")
        plt.legend()
        plt.savefig(save_loc_analysis + "/AllFramesForEachNodes" + ".png")
        plt.show()
        plt.cla()
        #plt.show()
        #pdb.set_trace()

    #def PlotTrajForSimNode(self,para_file, save_loc_analysis):


class LossAnalysis():
    def findClosestPointOnLine(self, sim_nodes, points_cloud, sim_num, gt_num):

        sim_nodes_diff = torch.diff(sim_nodes, dim=0)
        sim_nodes_diff_square = torch.linalg.norm(sim_nodes_diff, dim=1)**2
        closestpoint_factor = torch.sum(
            (points_cloud - sim_nodes[:-1]) * sim_nodes_diff,
            dim=2)  # factor to judge whether the projection point on the line
        closestpoint_factor = closestpoint_factor / sim_nodes_diff_square

        # if the projection of the point is not on the line, change the factor to ensure the projection on the line
        negative_index = torch.where(closestpoint_factor < 0)
        positive_index = torch.where(closestpoint_factor > 1)
        closestpoint_factor[negative_index] = 0
        closestpoint_factor[positive_index] = 1

        projection_points = sim_nodes[:
                                      -1] + closestpoint_factor[:, :,
                                                                None] * sim_nodes_diff[
                                                                    None, :, :]

        return projection_points

    ## ================================================================================
    #   Get the loss for line projection
    ## ================================================================================
    def getProjectionLineSegmentsLoss(self, sim_num, gt_num, sim_nodes,
                                      gt_points):

        # Note: this function is applicable to 2d and 3d centerline projection
        #pdb.set_trace()

        #pdb.set_trace()

        gt_points_repeat = gt_points.repeat(1, sim_num - 1)
        gt_points_repeat = gt_points_repeat.reshape(
            gt_num, sim_num - 1,
            sim_nodes.shape[1])  ## size : num_gt*(sim_num-1)*3

        proj_gt_points = self.findClosestPointOnLine(sim_nodes,
                                                     gt_points_repeat, sim_num,
                                                     gt_num)
        result = proj_gt_points - gt_points[:, None, :]
        distance = torch.linalg.norm(result, dim=2)
        min_index = torch.min(distance, dim=1)
        x_index = torch.linspace(0, gt_num - 1, gt_num)
        x_index = torch.as_tensor(x_index, dtype=torch.long)
        dmin = distance[x_index, min_index.indices]

        return torch.sum(dmin)

    ## ================================================================================

    #  Get corresponding point to point loss
    ## ================================================================================

    def getP2PCorrespondLoss(self,
                             gt_centerline,
                             node_pos_curr,
                             two_dimension=True):
        cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])
        if two_dimension:
            pt_3d_project = torch.matmul(cam_intrinsic_mat,
                                         node_pos_curr.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]
        else:
            sim_pbd = node_pos_curr

        real_gt = torch.as_tensor(gt_centerline)
        real_gt_diff = torch.diff(real_gt, dim=0).float()
        real_gt_diff_norm = torch.linalg.norm(real_gt_diff, dim=1)
        real_gt_length = torch.sum(real_gt_diff_norm)
        real_gt_normalized = real_gt_diff_norm / real_gt_length
        real_gt_norm_cum = torch.cumsum(real_gt_normalized, dim=0)
        real_gt_norm_cum = torch.hstack(
            (torch.as_tensor(0.0), real_gt_norm_cum)).float()

        sim_pbd_diff = torch.diff(sim_pbd, dim=0)
        sim_pbd_diff_norm = torch.linalg.norm(sim_pbd_diff, dim=1)
        sim_pbd_length = torch.sum(sim_pbd_diff_norm)
        sim_pbd_normalized = sim_pbd_diff_norm / sim_pbd_length
        sim_pbd_norm_cum = torch.cumsum(sim_pbd_normalized, dim=0)
        sim_pbd_norm_cum = torch.hstack(
            (torch.as_tensor(0.0), sim_pbd_norm_cum)).float()

        real_gt_norm_cum_extend = real_gt_norm_cum.repeat(
            sim_pbd_norm_cum.shape[0], 1).T
        real2sim_err = real_gt_norm_cum_extend[:] - sim_pbd_norm_cum
        real2sim_id = torch.min(abs(real2sim_err), 0)

        # real_gt_correspond_dist = real_gt_norm_cum[real2sim_id.indices]
        real_gt_correpond = real_gt[real2sim_id.indices]
        loss_correpond = torch.sum(
            torch.linalg.norm(sim_pbd - real_gt_correpond, dim=1))

        return loss_correpond, real2sim_id.indices

    ## ================================================================================

    #   Get the 2D projection point from 3D simulation point
    ## ================================================================================

    def get2DProjectionPointfrom3DSimPoints(self, sim_nodes):
        cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])
        projectpoint_2d_simnode = torch.matmul(cam_intrinsic_mat,
                                               sim_nodes.float().T).T
        projectpoint_2d_simnode_normalized = torch.div(
            projectpoint_2d_simnode,
            (projectpoint_2d_simnode[:, 2]).reshape(-1, 1))
        return projectpoint_2d_simnode_normalized[:, :2]

    ## ================================================================================
    #  Get segmentation projection loss for all the points
    ## ================================================================================
    def getSegProjectionLineLoss(self, gt_centerline, node_pos_curr,
                                 two_dimension):
        _, real2sim_id_indices = self.getP2PCorrespondLoss(
            gt_centerline, node_pos_curr, two_dimension)
        if two_dimension:
            sim_nodes = self.get2DProjectionPointfrom3DSimPoints(node_pos_curr)
        else:
            sim_nodes = node_pos_curr

        SegProjectionLineLoss = 0
        for i in range(sim_nodes.shape[0] - 1):
            #pdb.set_trace()
            curr_SimNode = sim_nodes[i:i + 2]
            curr_gt_points_index = real2sim_id_indices[i:i + 2]
            #pdb.set_trace()
            curr_gt_points = gt_centerline[
                curr_gt_points_index[0]:curr_gt_points_index[1]]
            SegProjectionLineLoss = SegProjectionLineLoss + self.getProjectionLineSegmentsLoss(
                curr_SimNode.shape[0], curr_gt_points.shape[0], curr_SimNode,
                curr_gt_points)
        return SegProjectionLineLoss

    ## ================================================================================
    #   Create Dataset for ErrorBar
    ## ================================================================================
    def ErrorBar(self):
        file_path = "../DATA/save_loss_withConstConvergency/"
        error_method_files = ["correspondance", "seg", "projection"]
        error_target_files = [
            "centerline_2d", "centerline_3d_gravity", "centerline_2d_plus_lwst"
        ]

        for _, error_method in enumerate(error_method_files):
            for _, error_target in enumerate(error_target_files):
                loss_list = []
                #pdb.set_trace()
                load_path = file_path + str(error_method) + "/" + str(
                    error_target)

                npy_files = listdir(load_path)

                for file in npy_files:

                    if os.path.splitext(file)[1] == ".npy":
                        if int(os.path.splitext(file)[0].split('_')[1]) < 30:
                            continue

                    num_frame = os.path.splitext(file)[0].split('_')[1]

                    sim_node_file_path = load_path + "/" + file
                    sim_nodes = torch.as_tensor(
                        np.load(sim_node_file_path)[:, :3])

                    centerline_3d_gravity = torch.as_tensor(
                        np.load(
                            "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                            % int(num_frame)))
                    #pdb.set_trace()
                    loss_3D = self.getProjectionLineSegmentsLoss(
                        sim_nodes.shape[0], centerline_3d_gravity.shape[0],
                        sim_nodes, centerline_3d_gravity)

                    loss_list.append(loss_3D.detach().numpy())
                loss_list = np.array(loss_list)
                #pdb.set_trace()
                np.save(
                    "../DATA/ErrorBar/" + error_method + "_" + error_target +
                    ".npy", loss_list)

    ## ================================================================================
    #   Create Dataset for 2D ErrorBar
    ## ================================================================================
    def ErrorBar2D(self):
        file_path = "../DATA/save_loss_withConstConvergency/"
        error_method_files = ["correspondance", "seg", "projection"]
        error_target_files = [
            "centerline_2d", "centerline_3d_gravity", "centerline_2d_plus_lwst"
        ]

        for _, error_method in enumerate(error_method_files):
            for _, error_target in enumerate(error_target_files):
                loss_list = []
                #pdb.set_trace()
                load_path = file_path + str(error_method) + "/" + str(
                    error_target)

                npy_files = listdir(load_path)

                for file in npy_files:

                    if os.path.splitext(file)[1] == ".npy":
                        if int(os.path.splitext(file)[0].split('_')[1]) < 30:
                            continue

                    num_frame = os.path.splitext(file)[0].split('_')[1]

                    sim_node_file_path = load_path + "/" + file
                    sim_nodes = torch.as_tensor(
                        np.load(sim_node_file_path)[:, :3])

                    centerline_2d_image = torch.as_tensor(
                        np.load(
                            "../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy"
                            % int(num_frame)))
                    #pdb.set_trace()
                    loss_segpro_line_2D = self.getSegProjectionLineLoss(
                        centerline_2d_image, sim_nodes,
                        two_dimension=True) / centerline_2d_image.shape[0]

                    loss_list.append(loss_segpro_line_2D.detach().numpy())
                loss_list = np.array(loss_list)
                #pdb.set_trace()
                np.save(
                    "../DATA/ErrorBar2D/" + error_method + "_" + error_target +
                    ".npy", loss_list)

    ## ================================================================================
    #  Plot vioin ErrorBar
    ## ================================================================================
    def PlotViolin(self):

        projection_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar/projection_centerline_3d_gravity.npy")
        #plt.violinplot(projection_centerline_3d_gravity_data)

        projection_centerline_2d_data = np.load(
            "../DATA/ErrorBar/projection_centerline_2d.npy")
        #plt.violinplot(projection_centerline_2d_data)

        projection_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar/projection_centerline_2d_plus_lwst.npy")
        #plt.violinplot(projection_centerline_2d_plus_lwst_data)

        correspondance_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar/correspondance_centerline_3d_gravity.npy")

        correspondance_centerline_2d_data = np.load(
            "../DATA/ErrorBar/correspondance_centerline_2d.npy")

        correspondance_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar/correspondance_centerline_2d_plus_lwst.npy")

        seg_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar/seg_centerline_3d_gravity.npy")

        seg_centerline_2d_data = np.load(
            "../DATA/ErrorBar/seg_centerline_2d.npy")

        seg_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar/seg_centerline_2d_plus_lwst.npy")

        # plt.set_title(' projection_centerline_3d_gravity')
        # plt.set_ylabel('Loss')
        # plt.violinplot(projection_centerline_3d_gravity_data)

        # plt.show()
        plt.style.use("seaborn")
        # stack data
        data = [
            projection_centerline_3d_gravity_data,
            projection_centerline_2d_data,
            projection_centerline_2d_plus_lwst_data,
            correspondance_centerline_3d_gravity_data,
            correspondance_centerline_2d_data,
            correspondance_centerline_2d_plus_lwst_data,
            seg_centerline_3d_gravity_data, seg_centerline_2d_data,
            seg_centerline_2d_plus_lwst_data
        ]
        #pdb.set_trace()
        label_data = np.array(data)
        # ------- Use different color to represent the result --------
        labels = []

        def add_label(violin, label):

            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        # -------- Length difference for Tomas --------
        def Execute_label(i, violin_data):
            positions = np.arange(i, i + 1, 1).astype(np.int)
            add_label(plt.violinplot(violin_data, positions), "OBJ" + str(i))

        for i in range(label_data.shape[0]):
            Execute_label(i + 1, label_data[i])

        # -------- Plot the percent for the plot --------

        # #pdb.set_trace()
        # plt.style.use('seaborn')
        # violin = plt.violinplot(dataset=data, showextrema=True)
        # for patch in violin['bodies']:
        #     patch.set_facecolor('#CCFFFF')
        #     patch.set_edgecolor('#CCFFFF')
        #     patch.set_alpha(1.0)

        for i, d in enumerate(data):
            min_value, quantile1, median, quantile3, max_value = np.percentile(
                d, [0, 25, 50, 75, 100])
            print(median)
            plt.scatter(i + 1, median, color='red', zorder=4)
            plt.vlines(i + 1,
                       quantile1,
                       quantile3,
                       colors='#66FFFF', #chnage color
                       lw=1,
                       zorder=5)
            plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)
            #pdb.set_trace()

        # plt.xticks(ticks=[1, 2, 3],
        #            labels=[
        #                'projection_centerline_3d_gravity',
        #                'projection_centerline_2d',
        #                'projection_centerline_2d_plus_lwst'
        #            ])

        plt.xticks(ticks=np.arange(1, 10, 1),
                   labels=[
                       'OBJ1', 'OBJ2', 'OBJ3', 'OBJ4', 'OBJ5', 'OBJ6', 'OBJ7',
                       'OBJ8', 'OBJ9'
                   ])
        #plt.legend(*zip(*labels), loc=2)

        plt.title("Error for 3D projection Loss")
        plt.savefig('../ALLResult/DATA/ViolionBarForLoss.png')
        plt.cla()

    ## ================================================================================
    #  Plot vioin ErrorBar for 2D environment
    ## ================================================================================
    def Plot2DViolin(self):

        projection_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar2D/projection_centerline_3d_gravity.npy")
        #plt.violinplot(projection_centerline_3d_gravity_data)

        projection_centerline_2d_data = np.load(
            "../DATA/ErrorBar2D/projection_centerline_2d.npy")
        #plt.violinplot(projection_centerline_2d_data)

        projection_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar2D/projection_centerline_2d_plus_lwst.npy")
        #plt.violinplot(projection_centerline_2d_plus_lwst_data)

        correspondance_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar2D/correspondance_centerline_3d_gravity.npy")

        correspondance_centerline_2d_data = np.load(
            "../DATA/ErrorBar2D/correspondance_centerline_2d.npy")

        correspondance_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar2D/correspondance_centerline_2d_plus_lwst.npy")

        seg_centerline_3d_gravity_data = np.load(
            "../DATA/ErrorBar2D/seg_centerline_3d_gravity.npy")

        seg_centerline_2d_data = np.load(
            "../DATA/ErrorBar2D/seg_centerline_2d.npy")

        seg_centerline_2d_plus_lwst_data = np.load(
            "../DATA/ErrorBar2D/seg_centerline_2d_plus_lwst.npy")

        # plt.set_title(' projection_centerline_3d_gravity')
        # plt.set_ylabel('Loss')
        # plt.violinplot(projection_centerline_3d_gravity_data)

        # plt.show()
        plt.style.use('seaborn')

        data = [
            projection_centerline_3d_gravity_data,
            projection_centerline_2d_data,
            projection_centerline_2d_plus_lwst_data,
            correspondance_centerline_3d_gravity_data,
            correspondance_centerline_2d_data,
            correspondance_centerline_2d_plus_lwst_data,
            seg_centerline_3d_gravity_data, seg_centerline_2d_data,
            seg_centerline_2d_plus_lwst_data
        ]
        #pdb.set_trace()
        violin = plt.violinplot(dataset=data, showextrema=True)
        for patch in violin['bodies']:
            patch.set_facecolor('#CCFFFF')
            patch.set_edgecolor('#CCFFFF')
            patch.set_alpha(1.0)

        for i, d in enumerate(data):
            min_value, quantile1, median, quantile3, max_value = np.percentile(
                d, [0, 25, 50, 75, 100])
            print(median)
            plt.scatter(i + 1, median, color='red', zorder=4)
            plt.vlines(i + 1,
                       quantile1,
                       quantile3,
                       colors='#66FFFF',
                       lw=1,
                       zorder=5)
            plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)
            #pdb.set_trace()

        # plt.xticks(ticks=[1, 2, 3],
        #            labels=[
        #                'projection_centerline_3d_gravity',
        #                'projection_centerline_2d',
        #                'projection_centerline_2d_plus_lwst'
        #            ])

        plt.xticks(ticks=np.arange(1, 10, 1),
                   labels=[
                       'OBG1', 'OBG2', 'OBG3', 'OBG4', 'OBG5', 'OBG6', 'OBG7',
                       'OBG8', 'OBG9'
                   ])
        plt.title("Error for 2D projection Loss")
        plt.savefig('../ALLResult/DATA/ViolionBarForLossErrorBar2D.png')
        plt.cla()

    ## ================================================================================
    #  Plot Constraint Satification For All Loss
    ## ================================================================================

    def PlotConstraintForAllLoss(self, save_file):

        file_path = "../DATA/save_loss_withConstConvergency/"
        error_method_files = ["projection", "correspondance", "seg"]
        #error_method_files = ["correspondance", "seg"]
        error_target_files = [
            "centerline_2d", "centerline_3d_gravity", "centerline_2d_plus_lwst"
        ]

        OBG_index = 0

        for _, error_method in enumerate(error_method_files):
            for _, error_target in enumerate(error_target_files):
                print(error_method, error_target)

                # -------- Plt Style --------
                plt.style.use('seaborn')
                # forward_sim_node = np.load(
                #     "../DATA/save_loss_withConstConvergency/Foward/projection/centerline_3d_gravity/frame_30.npy"
                # )
                # #pdb.set_trace()

                # plt.plot(
                #     np.arange(1, 20, 1).astype(dtype=np.str),
                #     forward_sim_node[:-1, 10].reshape(-1),
                #     label="C_strain" + " for frame " + str(30) +
                #     " forward process",
                # )

                #pdb.set_trace()
                load_path = file_path + str(error_method) + "/" + str(
                    error_target)

                for num_frame in [30, 37, 44, 51, 58]:

                    C_bendig = []
                    C_strain = []
                    C_dist = []

                    sim_node_file_path = load_path + "/frame_" + str(
                        num_frame) + ".npy"
                    sim_data = torch.as_tensor(np.load(sim_node_file_path))

                    # C_dist.append(torch.mean(sim_data[:, 8]))
                    # C_bendig.append(torch.mean(sim_data[:, 9]))
                    # C_strain.append(torch.mean(sim_data[:, 10]))
                    C_dist.append(sim_data[:, 8].detach().numpy())
                    C_bendig.append(sim_data[:, 9].detach().numpy())
                    C_strain.append(sim_data[:, 10].detach().numpy())
                    C_dist = np.array(C_dist)
                    C_bendig = np.array(C_bendig)
                    C_strain = np.array(C_strain)

                    #print(plt.style.available)
                    plt.rcParams.update({'font.size': 12})

                    # -------- Plot constraint for Tomas Distance --------
                    # plt.title(
                    #     "Tomas Distance Constrain Satisfication for OBG%d" %
                    #     OBG_index)
                    # plt.plot(
                    #     np.arange(19).astype(dtype=np.str),
                    #     C_dist[0, :-1].reshape(-1),
                    #     label="C_dist" + " for frame " + str(num_frame),
                    # )

                    #-------- Plot constraint for Bending  --------
                    plt.title("Bending Constrain Satisfication for OBG%d" %
                              OBG_index)
                    plt.plot(
                        np.arange(1, 19, 1).astype(dtype=np.str),
                        C_bendig[0, :-2].reshape(-1),
                        label="C_bendig" + " for frame " + str(num_frame),
                    )

                    # -------- Plot constraint for Strain  --------
                    # plt.title("Strain Constrain Satisfication for OBG%d" %
                    #           OBG_index)
                    # plt.plot(
                    #     np.arange(1, 20, 1).astype(dtype=np.str),
                    #     C_strain[0, :-1].reshape(-1),
                    #     label="C_strain" + " for frame " + str(num_frame),
                    # )
                    plt.legend()

                    #pdb.set_trace()
                plt.ylabel("Constraint")
                plt.xlabel("frame")

                plt.savefig(save_file + "/Bending_" + error_method + "_" +
                            error_target + ".png")
                #plt.show()
                plt.cla()

                #pdb.set_trace()
                OBG_index += 1


## ================================================================================
#   Plot PS result
## ================================================================================
class PSSimNode():
    def ProjectionPtfrom3DNode(self, SimNode):
        #The cammer matrix, which can not be changed
        cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])

        pt_3d_project = torch.matmul(cam_intrinsic_mat, SimNode.float().T).T
        sim_pt_2d = torch.div(pt_3d_project, (pt_3d_project[:,
                                                            2]).reshape(-1, 1))
        SimNode_2D = sim_pt_2d[:, :2]
        return SimNode_2D

    def PlotPSSimNode(self, ps_file_path, frames_num, loss_file, save_path,
                      cal_loss_method, cal_loss_target):

        for frame_num in frames_num:

            # -------- Load simulation nodes and make 2D projectio --------
            sim_node = torch.as_tensor(
                np.load(loss_file + "/" + "frame_%d.npy" % frame_num))[:, :3]
            SimNode_2D = self.ProjectionPtfrom3DNode(sim_node)

            plt.plot(SimNode_2D[:, 0], SimNode_2D[:, 1])
            plt.scatter(SimNode_2D[:, 0], SimNode_2D[:, 1])

        RgbImage = plt.imread(ps_file_path)
        plt.imshow(RgbImage)

        plt.axis("off")

        plt.savefig(save_path + "/" + cal_loss_method + "_" + cal_loss_target +
                    "_" + str(frames_num) + ".png")
        plt.cla()


# -------- Find the Best parameter -------
class FindBestParameter():
    def BestParameter(self):
        file_path = "../DATA/optimal_para_data/XPBD_54/"
        loss = []
        for i in range(100):
            file_path_location = file_path + "frame_54_itr_" + str(i) + ".npy"
            data = np.load(file_path_location)

            loss.append(data[6, 0])
        plt.plot(np.arange(len(loss)), loss)
        min_loss_index = np.argmin(loss)

        #------- The weigth with minimum loss -------
        fmin_loss_path_location = file_path + "frame_54_itr_" + str(
            min_loss_index) + ".npy"
        data = np.load(fmin_loss_path_location)
        print(loss[min_loss_index])
        print(data[0:7, 0])

        pdb.set_trace()


'''

XPBD 30 best parameter
self.w_gravity = 0.26472151

self.w_dis = 1.19192898

self.w_strain = 0.80006707
# effect : if on the same plane

self.wq_strain = 1.19918275

self.wq_bending = 0.80061877

# effect : curvature
self.w_SOR = 0.79988092
'''
'''

XPBD 54 best parameter
self.w_gravity = 0.04928657

self.w_dis = 1.04609156

self.w_strain = 1.05289471
# effect : if on the same plane

self.wq_strain =1.05171621

self.wq_bending = 1.05238736

# effect : curvature
self.w_SOR = 0.79988092
'''


# -------- Make a comparision over the 2D and 3D raw centerline  data ----------
class CompareRawData():
    def compare(self):
        for num_frame in range(30, 66):
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % num_frame))

            centerline_2d_image = np.load(
                "../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy" %
                num_frame)

            cam_intrinsic_mat = torch.as_tensor(
                [[960.41357421875, 0.0, 1021.7171020507812],
                 [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])

            pt_3d_project = torch.matmul(cam_intrinsic_mat,
                                         centerline_3d_gravity.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]
            plt.plot(sim_pbd[:, 0], sim_pbd[:, 1])
            plt.plot(centerline_2d_image[:, 0], centerline_2d_image[:, 1])
            plt.show()
            pdb.set_trace()
            plt.cla()


# -------- Make a comparision over the 2D and 3D raw centerline  data ----------
class CompareOverXPBDAndTomas():
    def compare(self, node_file, save_name, frames, dist, bending, strain):

        sns.set(style="darkgrid")

        # -------- Set the title for the plot -------
        if dist == True:
            plt.title("C_dist for " + save_name)
        if bending == True:
            plt.title("C_bending for " + save_name)
        if strain == True:
            plt.title("C_strain for " + save_name)

        for num_frame in frames:

            sim_nodes_data = np.load(node_file + "/frame_%d.npy" % num_frame)

            C_dist = sim_nodes_data[:, 8]
            C_bending = sim_nodes_data[:, 9]
            C_strain = sim_nodes_data[:, 10]

            # ------- Set the plotting data --------
            if dist == True:
                frame = list(np.arange(19).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_dist[:-1]))
            if bending == True:
                frame = list(np.arange(18).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_bending[:-2]))

            if strain == True:
                frame = list(np.arange(19).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_strain[:-1]))
            #pdb.set_trace()

            df = pd.DataFrame(data=sns_dataset, columns=['Node', 'C_dist'])
            ax = sns.lineplot(data=df,
                              x="Node",
                              y="C_dist",
                              label="Frame" + str(num_frame))

            sns.scatterplot(
                data=df,
                x="Node",
                y="C_dist",
            )

        # ------ Set the fig name --------
        if dist == True:
            plt.savefig("./DATA/" + save_name + "_dist.png")
        if bending == True:
            plt.savefig("./DATA/" + save_name + "_bending.png")
        if strain == True:
            plt.savefig("./DATA/" + save_name + "_strain.png")
        plt.cla()
        #pdb.set_trace()

    def CompareTogether(self, frames, dist, bending, strain):

        files_path = [
            "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/",
            "../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/"
        ]

        labels = np.array(["Tomas", "GS XPBD"])

        sns.set(style="darkgrid")
        # -------- Set the title for the plot -------
        if dist == True:
            plt.title("C_dist for Tomas VS GS XPBD")
        if bending == True:
            plt.title("C_bending for Tomas VS GS XPBD ")
        if strain == True:
            plt.title("C_strain for Tomas VS GS XPBD ")

        for index, node_file in enumerate(files_path):
            for num_frame in frames:
                sim_nodes_data = np.load(node_file +
                                         "/frame_%d.npy" % num_frame)

                C_dist = sim_nodes_data[:, 8]
                C_bending = sim_nodes_data[:, 9]
                C_strain = sim_nodes_data[:, 10]

                # ------- Set the plotting data --------
                if dist == True:
                    frame = list(np.arange(19).astype(dtype=np.str))
                    sns_dataset = list(zip(frame, C_dist[:-1]))
                if bending == True:
                    frame = list(np.arange(18).astype(dtype=np.str))
                    sns_dataset = list(zip(frame, C_bending[:-2]))

                if strain == True:
                    frame = list(np.arange(19).astype(dtype=np.str))
                    sns_dataset = list(zip(frame, C_strain[:-1]))
                    #pdb.set_trace()

                # ------- Plot Data --------
                if dist == True:
                    df = pd.DataFrame(data=sns_dataset,
                                      columns=['Node', 'C_dist'])

                    ax = sns.lineplot(data=df,
                                      x="Node",
                                      y="C_dist",
                                      label="Frame " + str(num_frame) +
                                      " for " + labels[index])
                    sns.scatterplot(
                        data=df,
                        x="Node",
                        y="C_dist",
                    )

                if bending == True:
                    df = pd.DataFrame(data=sns_dataset,
                                      columns=['Node', 'C_bending'])

                    ax = sns.lineplot(data=df,
                                      x="Node",
                                      y="C_bending",
                                      label="Frame " + str(num_frame) +
                                      " for " + labels[index])
                    sns.scatterplot(
                        data=df,
                        x="Node",
                        y="C_bending",
                    )

                if strain == True:
                    df = pd.DataFrame(data=sns_dataset,
                                      columns=['Node', 'C_strain'])

                    ax = sns.lineplot(data=df,
                                      x="Node",
                                      y="C_strain",
                                      label="Frame " + str(num_frame) +
                                      " for " + labels[index])
                    sns.scatterplot(
                        data=df,
                        x="Node",
                        y="C_strain",
                    )
        # ------ Set the fig name --------
        if dist == True:
            plt.savefig("./DATA/" + "TomasVSXPBD_curve" + "_dist.png")
        if bending == True:
            plt.savefig("./DATA/" + "TomasVSXPBD_curve" + "_bending.png")
        if strain == True:
            plt.savefig("./DATA/" + "TomasVSXPBD_curve" + "_strain.png")

        plt.cla()

    # -------- Plot a plot in a plot -------
    def PlotInPlot(self, node_file, save_name, frames, dist, bending, strain):

        sns.set(style="darkgrid")
        fig = plt.figure()

        # -------- Set the title for the plot -------
        if dist == True:
            sim_nodes_data = np.load(node_file + "/frame_%d.npy" % num_frame)

            C_dist = sim_nodes_data[:, 8]
            C_bending = sim_nodes_data[:, 9]
            C_strain = sim_nodes_data[:, 10]

            # ------- Set the plotting data --------
            if dist == True:
                frame = list(np.arange(19).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_dist[:-1]))
            if bending == True:
                frame = list(np.arange(18).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_bending[:-2]))

            if strain == True:
                frame = list(np.arange(19).astype(dtype=np.str))
                sns_dataset = list(zip(frame, C_strain[:-1]))
            #pdb.set_trace()

            df = pd.DataFrame(data=sns_dataset, columns=['Node', 'C_dist'])
            ax = sns.lineplot(data=df,
                              x="Node",
                              y="C_dist",
                              label="Frame" + str(num_frame))

        # ------ Set the fig name --------
        if dist == True:
            plt.savefig("./DATA/" + save_name + "_dist.png")
        if bending == True:
            plt.savefig("./DATA/" + save_name + "_bending.png")
        if strain == True:
            plt.savefig("./DATA/" + save_name + "_strain.png")

        x = [1, 2, 3, 4, 5, 6, 7]
        y = [1, 3, 4, 2, 5, 8, 6]
        left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
        ax1 = fig.add_axes([left, bottom, width, height])
        RGBIMAGE = plt.imread(
            "../DATA/RealtoSimPlotting/XPBD/projection/centerline_3d_gravity_curve/50.png"
        )
        ax1.imshow(RGBIMAGE)

    def CompareLength(self):
        correct_frame = torch.as_tensor(
            np.load(
                "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_14_init.npy"
            ))[:, :3]
        correct_sim_node_diff = torch.diff(correct_frame, dim=0)
        correct_lenght = torch.sum(correct_sim_node_diff).detach().numpy()

        Tomas_lengths = []
        XPBD_lengths = []
        centerline_lengths = []

        Tomas_control = []
        XPBD_control = []
        centerline_control = []
        for num_frame in range(30, 66):
            # -------- Information for sim nodes from Tomas --------
            Tomas_sim_nodes = torch.as_tensor(
                np.load(
                    "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                    % num_frame))[:, :3]
            Tomas_sim_nodes_diff = torch.diff(Tomas_sim_nodes, dim=0)
            Tomas_length = torch.sum(Tomas_sim_nodes_diff)
            Tomas_lengths.append(Tomas_length)
            Tomas_control.append(Tomas_sim_nodes[0].detach().numpy())

            # -------- Information for sim nodes for XPBD --------
            XPBD_sim_nodes = torch.as_tensor(
                np.load(
                    "../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/frame_%d.npy"
                    % num_frame))[:, :3]
            XPBD_sim_nodes_diff = torch.diff(XPBD_sim_nodes, dim=0)
            XPBD_length = torch.sum(XPBD_sim_nodes_diff)
            XPBD_lengths.append(XPBD_length)
            XPBD_control.append(XPBD_sim_nodes[0].detach().numpy())

            # -------- Information for centerline--------
            centerline_3d = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % num_frame))[:, :3]
            centerline_3d_diff = torch.diff(centerline_3d, dim=0)
            centerline_length = torch.sum(centerline_3d_diff)
            centerline_lengths.append(centerline_length)
            centerline_control.append(centerline_3d[0].detach().numpy())

        Tomas_lengths = np.array(Tomas_lengths) / np.array(
            centerline_lengths) * correct_lenght
        XPBD_lengths = np.array(XPBD_lengths) / np.array(
            centerline_lengths) * correct_lenght
        centerline_lengths = np.array(centerline_lengths) / np.array(
            centerline_lengths) * correct_lenght

        plt.style.use('seaborn')
        plt.xlim(30, 66)

        # -------- Pack and plot XPBD Data --------
        frame = np.arange(30, 66, 1)
        plt.plot(frame, XPBD_lengths, label="Length for XPBD")
        plt.scatter(frame, XPBD_lengths, s=20)

        # -------- Pack and plot Tomas Data --------
        frame = np.arange(30, 66, 1)
        plt.plot(frame, Tomas_lengths, label="Length for Tomas")
        plt.scatter(frame, Tomas_lengths, s=20)

        # -------- Pack and plot centerline Data --------

        centerline_lengths_dataset = list(zip(frame, centerline_lengths))
        # df = pd.DataFrame(data=centerline_lengths_dataset,
        #                   columns=['Frames', 'Length'])
        plt.plot(frame, centerline_lengths, label="Length for centerline")
        plt.scatter(frame, centerline_lengths, s=20)

        plt.xlabel("Frame")
        plt.ylabel("Length")
        plt.legend()
        plt.savefig("./DATA/Length_compare.png")

        plt.cla()

        # -------- Plot lenght loss -------
        data = [Tomas_lengths - correct_lenght, XPBD_lengths - correct_lenght]
        violin = plt.violinplot(dataset=data, showextrema=True)
        for patch in violin['bodies']:
            patch.set_facecolor('#CCFFFF')
            patch.set_edgecolor('#CCFFFF')
            patch.set_alpha(1.0)

        for i, d in enumerate(data):
            min_value, quantile1, median, quantile3, max_value = np.percentile(
                d, [0, 25, 50, 75, 100])

            plt.scatter(i + 1, median, color='red', zorder=4)
            plt.vlines(i + 1,
                       quantile1,
                       quantile3,
                       colors='#66FFFF',
                       lw=1,
                       zorder=5)
            plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)

        plt.xticks(ticks=np.arange(1, 3, 1),
                   labels=[
                       "Length difference for Tomas",
                       "Length difference for XPBD",
                   ])

        plt.savefig("./DATA/length_loss.png")
        plt.cla()

        # --------control point loss --------
        #pdb.set_trace()

        Tomas_control = np.array(Tomas_control)
        XPBD_control = np.array(XPBD_control)
        centerline_control = np.array(centerline_control)

        Tomas_control_loss = np.linalg.norm(Tomas_control - centerline_control,
                                            axis=1)

        XPBD_control_loss = np.linalg.norm(XPBD_control - centerline_control,
                                           axis=1)

        frame = np.arange(30, 66, 1)

        # -------- Plot Tomas Traj loss -------
        plt.style.use('seaborn')
        plt.plot(frame,
                 Tomas_control_loss,
                 label="Control Point Loss for Tomas")
        plt.scatter(frame, Tomas_control_loss, s=20)

        # -------- Plot XPBD Traj loss -------
        plt.plot(frame, XPBD_control_loss, label="Control Point Loss for XPBD")
        plt.scatter(frame, XPBD_control_loss, s=20)

        plt.ylabel("Control Point loss")
        plt.xlabel("Frame")
        plt.legend()

        #plt.show()
        plt.savefig("./DATA/traj_loss.png")
        plt.cla()

        # -------- Plot Violin Loss  --------

        data = [Tomas_control_loss, XPBD_control_loss]
        violin = plt.violinplot(dataset=data, showextrema=True)
        for patch in violin['bodies']:
            patch.set_facecolor('#CCFFFF')
            patch.set_edgecolor('#CCFFFF')
            patch.set_alpha(1.0)

        for i, d in enumerate(data):
            min_value, quantile1, median, quantile3, max_value = np.percentile(
                d, [0, 25, 50, 75, 100])

            plt.scatter(i + 1, median, color='red', zorder=4)
            plt.vlines(i + 1,
                       quantile1,
                       quantile3,
                       colors='#66FFFF',
                       lw=1,
                       zorder=5)
            plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)

        plt.xticks(ticks=np.arange(1, 3, 1),
                   labels=[
                       "Traj difference for Tomas",
                       "Traj difference for XPBD",
                   ])

        plt.savefig("./DATA/traj_loss_violin.png")
        plt.cla()

        return Tomas_control_loss, XPBD_control_loss, Tomas_lengths - correct_lenght, XPBD_lengths - correct_lenght, Tomas_control_loss, XPBD_control_loss

        #--------

        #pdb.set_trace()
    def CompareTraj(self):

        fig_curve = plt.figure()
        ax_curve = Axes3D(fig_curve)

        Tomas_control_point = []
        XPBD_control_point = []
        True_control_point = []
        for num_frame in range(30, 66):
            #-------- Sim node from Tomas -------

            Tomas_sim_node = np.load(
                "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                % num_frame)[:, :3]
            Tomas_control_point.append(Tomas_sim_node[0])

            #-------- Sim node from XPBD -------
            XPBD_sim_node = np.load(
                "../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/frame_%d.npy"
                % num_frame)[:, :3]
            XPBD_control_point.append(XPBD_sim_node[0])

            #-------- Sim node from Gound True -------
            Ground_True_control_point = np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)[:, :3]
            True_control_point.append(Ground_True_control_point[0])

        Tomas_control_point = np.array(Tomas_control_point)
        XPBD_control_point = np.array(XPBD_control_point)
        True_control_point = np.array(True_control_point)
        ax_curve.plot3D(Tomas_control_point[:, 0], Tomas_control_point[:, 1],
                        Tomas_control_point[:, 2])
        # ax_curve.plot3D(XPBD_control_point[:, 0], XPBD_control_point[:, 1],
        #                 XPBD_control_point[:, 2])
        ax_curve.plot3D(True_control_point[:, 0], True_control_point[:, 1],
                        True_control_point[:, 2])

        pdb.set_trace()

    def CompareConstrainSatisificationForTomasAndXPBD(self):

        C_dist_Tomas = []
        C_bending_Tomas = []
        C_strain_Tomas = []
        C_dist_XPBD = []
        C_bending_XPBD = []
        C_strain_XPBD = []
        loss = []

        for num_frame in range(30, 66):
            # -------- Save Constraint Information for Tomas --------
            sim_node_Tomas = np.load(
                "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                % num_frame)
            C_dist_Tomas.append(np.max(sim_node_Tomas[:, 8]))
            C_bending_Tomas.append(np.max(sim_node_Tomas[:, 9]))
            C_strain_Tomas.append(np.max(sim_node_Tomas[:, 10]))

            # -------- Save Constraint Information for XPBD --------
            sim_node_XPBD = np.load(
                "../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/frame_%d.npy"
                % num_frame)
            C_dist_XPBD.append(np.max(sim_node_XPBD[:, 8]))
            C_bending_XPBD.append(np.max(sim_node_XPBD[:, 9]))
            C_strain_XPBD.append(np.max(sim_node_XPBD[:, 10]))

        plt.style.use('seaborn')

        # -------- Plot Constraint Information for Tomas -------
        plt.plot(np.arange(30, 66, 1),
                 np.array(C_dist_Tomas).reshape(-1),
                 label="C_dist for Tomas")
        # plt.plot(np.arange(len(C_bending_Tomas)),
        #          np.array(C_bending_Tomas).reshape(-1),
        #          label="C_bending for Tomas")
        # plt.plot(np.arange(len(C_strain_Tomas)),
        #          np.array(C_strain_Tomas).reshape(-1),
        #          label="C_strain for Tomas")

        plt.scatter(
            np.arange(30, 66, 1),
            np.array(C_dist_Tomas).reshape(-1),
        )
        # plt.scatter(
        #     np.arange(len(C_bending_Tomas)),
        #     np.array(C_bending_Tomas).reshape(-1),
        # )
        # plt.scatter(
        #     np.arange(len(C_strain_Tomas)),
        #     np.array(C_strain_Tomas).reshape(-1),
        # )

        # -------- Plot Constraint Information for XPBD -------
        plt.plot(np.arange(30, 66, 1),
                 np.array(C_dist_XPBD).reshape(-1),
                 label="C_dist for XPBD")
        # plt.plot(np.arange(len(C_bending_XPBD)),
        #          np.array(C_bending_XPBD).reshape(-1),
        #          label="C_bending for XPBD")
        # plt.plot(np.arange(len(C_strain_XPBD)),
        #          np.array(C_strain_XPBD).reshape(-1),
        #          label="C_strain for XPBD")

        plt.scatter(
            np.arange(30, 66, 1),
            np.array(C_dist_XPBD).reshape(-1),
        )
        # plt.scatter(
        #     np.arange(len(C_bending_XPBD)),
        #     np.array(C_bending_XPBD).reshape(-1),
        # )
        # plt.scatter(
        #     np.arange(len(C_strain_XPBD)),
        #     np.array(C_strain_XPBD).reshape(-1),
        # )
        plt.xlabel("Frame")
        plt.ylabel("Constraint")

        plt.legend()
        plt.savefig("./DATA/PlotSimNodeResultAnalysisALLFrames.png")
        plt.cla()
        #plt.show()
        #pdb.set_trace()
        return C_dist_Tomas, C_dist_XPBD

    def CompareCDistAndTrajLoss(self):
        plt.style.use('seaborn')

        # -------------------------------------
        # -------- Plot Traj loss -------------
        # -------------------------------------
        plt.subplot(3, 1, 1)
        Tomas_control_loss, XPBD_control_loss, _, _, _, _ = self.CompareLength(
        )
        plt.cla()

        frame = np.arange(30, 66, 1)
        plt.plot(frame,
                 Tomas_control_loss,
                 label="Control Point Loss for Tomas")
        plt.scatter(frame, Tomas_control_loss, s=20)

        plt.plot(
            frame,
            0 * frame,
            label="Excepted Control Point Loss",
            linestyle='--',
        )

        # -------- Plot XPBD Traj loss -------
        plt.plot(frame, XPBD_control_loss, label="Control Point Loss for XPBD")
        plt.scatter(frame, XPBD_control_loss, s=20)
        plt.title("Control Point Loss")
        plt.ylabel("Loss(m)")

        plt.legend()

        # -------------------------------------
        # -------- Plot C_dist ----------------
        # -------------------------------------
        plt.subplot(3, 1, 2)
        # -------- C_dist -------
        C_dist_Tomas, C_dist_XPBD = self.CompareConstrainSatisificationForTomasAndXPBD(
        )

        # -------- Plot Tomas Traj loss -------
        plt.plot(frame, C_dist_Tomas, label="C_dist Tomas")
        plt.scatter(frame, C_dist_Tomas, s=20)

        # -------- Plot XPBD Traj loss -------

        plt.plot(frame, C_dist_XPBD, label="C_dist for XPBD")
        plt.scatter(frame, C_dist_XPBD, s=20)

        plt.plot(
            frame,
            0 * frame,
            label="Excepted C_dist",
            linestyle='--',
        )
        plt.ylabel("Loss(m)")
        plt.title("C_dist")
        plt.legend()

        # -------------------------------------
        # -------- Plot Length ----------------
        # -------------------------------------
        plt.subplot(3, 1, 3)

        compare = CompareOverXPBDAndTomas()
        _, _, Tomas_legnths, XPBD_lengths, _, _ = compare.CompareLength()
        plt.cla()
        Tomas_legnths = Tomas_legnths + 0.8
        XPBD_lengths = XPBD_lengths + 0.8

        plt.style.use('seaborn')
        plt.xlim(30, 66)
        centerline_lengths = np.ones((Tomas_legnths.shape))
        centerline_lengths[:] = 0.8006

        # -------- Pack and plot XPBD Data --------
        frame = np.arange(30, 66, 1)
        plt.plot(frame, XPBD_lengths, label="Length for XPBD")
        plt.scatter(frame, XPBD_lengths, s=20)

        # -------- Pack and plot Tomas Data --------
        frame = np.arange(30, 66, 1)
        plt.plot(frame, Tomas_legnths, label="Length for Tomas")
        plt.scatter(frame, Tomas_legnths, s=20)

        # -------- Pack and plot centerline Data --------

        # df = pd.DataFrame(data=centerline_lengths_dataset,
        #                   columns=['Frames', 'Length'])
        plt.plot(
            frame,
            centerline_lengths,
            label="Length for centerline",
            linestyle='--',
        )
        #plt.scatter(frame, centerline_lengths, s=20)
        plt.title("Length")
        plt.ylabel("Length(m)")

        plt.legend()
        plt.savefig("./DATA/CompareCDistAndTrajLoss.png")
        plt.cla()

        #plt.show()
        #pdb.set_trace()

    def CombineTrajLossAndLengthLoss(self):
        _, _, Tomas_lengths_Loss, XPBD_lengths_Loss, Tomas_control_loss, XPBD_control_loss = self.CompareLength(
        )
        plt.cla()

        labels = []

        def add_label(violin, label):

            color = violin["bodies"][0].get_facecolor().flatten()
            labels.append((mpatches.Patch(color=color), label))

        # -------- Length difference for Tomas --------
        positions = np.arange(1, 2, 1).astype(np.int)
        data = Tomas_lengths_Loss
        add_label(plt.violinplot(data, positions),
                  "Length difference for Tomas")

        # -------- Length difference for XPBD --------
        positions = np.arange(2, 3, 1).astype(np.int)
        data = XPBD_lengths_Loss
        add_label(plt.violinplot(data, positions),
                  "Length difference for XPBD")

        # -------- Control Point difference for Tomas --------
        positions = np.arange(3, 4, 1).astype(np.int)
        data = Tomas_control_loss
        add_label(plt.violinplot(data, positions),
                  "Control Point difference for Tomas")

        # -------- Control Point difference for XPBD --------
        positions = np.arange(4, 5, 1).astype(np.int)
        data = XPBD_control_loss
        add_label(plt.violinplot(data, positions),
                  "Control Point difference for XPBD")

        # ------- Plot the percent for the error --------
        data = [
            Tomas_lengths_Loss, XPBD_lengths_Loss, Tomas_control_loss,
            XPBD_control_loss
        ]

        for i, d in enumerate(data):
            min_value, quantile1, median, quantile3, max_value = np.percentile(
                d, [0, 25, 50, 75, 100])

            plt.scatter(i + 1, median, color='red', zorder=4)
            plt.vlines(i + 1,
                       quantile1,
                       quantile3,
                       colors='#66FFFF',
                       lw=1,
                       zorder=5)
            plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)

        plt.legend(*zip(*labels), loc=2)
        plt.ylim(0, 1.05)
        plt.xticks(ticks=np.arange(1, 5, 1),
                   labels=[
                       "Length ",
                       "Length",
                       "Control Point ",
                       "Control Point ",
                   ])
        plt.savefig("../DATA/CombineTrajLossAndLengthLoss.png")
        #plt.show()
        #pdb.set_trace()

        # -------- Plot lenght Loss and control Loss together --------

        # data = [
        #     Tomas_lengths_Loss, XPBD_lengths_Loss, Tomas_control_loss,
        #     XPBD_control_loss
        # ]
        # violin = plt.violinplot(dataset=data, showextrema=True)
        # for patch in violin['bodies']:
        #     patch.set_facecolor('#CCFFFF')
        #     patch.set_edgecolor('#CCFFFF')
        #     patch.set_alpha(1.0)

        # for i, d in enumerate(data):
        #     min_value, quantile1, median, quantile3, max_value = np.percentile(
        #         d, [0, 25, 50, 75, 100])

        #     plt.scatter(i + 1, median, color='red', zorder=4)
        #     plt.vlines(i + 1,
        #                quantile1,
        #                quantile3,
        #                colors='#66FFFF',
        #                lw=1,
        #                zorder=5)
        #     plt.vlines(i + 1, min_value, max_value, colors='b', zorder=10)

        # plt.xticks(ticks=np.arange(1, 5, 1),
        #            labels=[
        #                "Length difference for Tomas",
        #                "Length difference for XPBD",
        #                "Traj difference for Tomas",
        #                "Traj difference for XPBD",
        #            ])

        plt.savefig("./DATA/CombineTrajLossAndLengthLoss.png")
        plt.cla()


## ================================================================================
#   Plot Forward Process Result
## ================================================================================
class ForwardPlot():
    def PlotForwardXPBD(self):

        plt.style.use('seaborn')

        # --------Correct Lenght Calculation --------
        correct_frame = torch.as_tensor(
            np.load(
                "../DATA/DATA_BaxterRope/downsampled_pcl/initialization.npy")
        )[:, :3]

        correct_lenght = torch.linalg.norm(correct_frame[0] -
                                           correct_frame[-1]).detach().numpy()
        correct_lengths = np.array([0.7782, 0.8006])

        for index, frame_num in enumerate([30,54]):

            frame_data = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % frame_num))[:, :3]
            frame_data_diff = torch.diff(frame_data, dim=0)
            frame_lenght = torch.sum(frame_data_diff).detach().numpy()

            length_XPBD = []
            length_Tomas = []
            C_dist_XPBD = []
            C_dist_Tomas = []

            for itr in range(30):
                for step in range(30):

                    # ------- XPBD simulation data --------
                    sim_nodes_XPBD_data = torch.as_tensor(
                        np.load("../DATA/CompareOverXPBDAndTomas/XPBD_" +
                                str(frame_num) + "/" + str(itr) + "_" +
                                str(step) + ".npy"))

                    sim_nodes_XPBD = sim_nodes_XPBD_data[:, :3]
                    sim_nodes_XPBD_diff = torch.diff(sim_nodes_XPBD, dim=0)
                    # sim_nodes_XPBD_diff_norm = torch.linalg.norm(
                    #     sim_nodes_XPBD_diff, dim=1)

                    length_XPBD.append(sim_nodes_XPBD_data[-1, 3])
                    C_dist_XPBD.append(torch.max(sim_nodes_XPBD_data[:-1, 3]))

                    # ------- Tomas simulation data --------
                    sim_nodes_Tomas_data = torch.as_tensor(
                        np.load("../DATA/CompareOverXPBDAndTomas/Tomas_" +
                                str(frame_num) + "/" + str(itr) + "_" +
                                str(step) + ".npy"))
                    #pdb.set_trace()

                    sim_nodes_Tomas = sim_nodes_Tomas_data[:-1, :3]
                    sim_nodes_Tomas_diff = torch.diff(sim_nodes_Tomas, dim=0)
                    # sim_nodes_Tomas_diff_diff_norm = torch.linalg.norm(
                    #     sim_nodes_Tomas_diff, dim=1)

                    # length_Tomas.append(
                    #     torch.sum(
                    #         sim_nodes_Tomas_diff_diff_norm).detach().numpy())
                    length_Tomas.append(sim_nodes_Tomas_data[-1, 3])

                    C_dist_Tomas.append(torch.max(sim_nodes_Tomas_data[:-1,
                                                                       3]))

                    #plt.plot(np.arange(19), sim_nodes_diff_norm)

            length_XPBD = np.array(length_XPBD)
            length_Tomas = np.array(length_Tomas)
            
            plt.subplot(1, 2, 1)
            #-------- plot length information for Tomas and XPBD --------
            print("Length for XPBD for frame " + str(frame_num))
            print(length_Tomas[10])

            plt.xlabel("Iteration")
            plt.ylabel("Length")
            plt.title("Length")
            plt.plot(np.arange(0, 10, 10 / 300),
                     length_XPBD[:300] / correct_lengths[index] *
                     correct_lenght,
                     label="Length for XPBD for frame " + str(frame_num))
            # plt.plot(np.arange(0, 10, 10 / 300),
            #          length_Tomas[:300] / correct_lengths[index] *
            #          correct_lenght,
            #          label="Length for Tomas for frame " + str(frame_num))
            plt.plot(np.arange(0, 10, 10 / 300),
                     length_Tomas[:300] ,
                     label="Length for Tomas for frame " + str(frame_num))
            plt.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.8))
           

            # -------- plot length constraint information for Tomas and XPBD --------
            plt.subplot(1, 2, 2)
            plt.xlabel("Iteration")
            plt.ylabel("Constraint")
            plt.title("C_dist")
            plt.plot(np.arange(0, 10, 10 / 300),
                     C_dist_XPBD[:300],
                     label="C_dist for XPBD for frame " + str(frame_num))
            plt.plot(np.arange(0, 10, 10 / 300),
                     C_dist_Tomas[:300],
                     label="C_dist for Tomas for frame " + str(frame_num))
            plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.1))
        

        #-------- Ground Truth length --------
        plt.subplot(1, 2, 1)
        plt.title("Length")
        plt.xlabel("Iteration")
        plt.ylabel("Length")
        correct_lenght = 0.78
        print("correct_lenght",correct_lenght)
        length = np.ones((300)) * correct_lenght
        plt.plot(np.arange(0, 10, 10 / 300),
                 length,
                 label="Ground Truth Length",
                 linestyle='--')
        plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))

        # -------- Ground True C_dist -------
        plt.subplot(1, 2, 2)
      
     
        C_dist = np.ones((300)) * 0
        plt.plot(np.arange(0, 10, 10 / 300),
                 C_dist,
                 label="Expected C_dist",
                 linestyle='--')

        plt.legend(loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
        plt.show()
        # plt.cla()

        # # -------- Length over iteration --------
        # plt.subplot(2,1,1)
        # plt.xlabel("Iteration")
        # plt.ylabel("Length")
        # plt.savefig("./DATA/length_forward_Tomas_XPBD.png")
        
        # plt.subplot(2,1,1)
        # plt.xlabel("Iteration")
        # plt.ylabel("Constraint")
        # plt.savefig("./DATA/C_dist_forward_Tomas_XPBD.png")

        plt.savefig("./DATA_ConnectImage/FowardTogetherLengthCdist.png")

        #pdb.set_trace()
        #plt.show()


from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import splrep, splev


class VisualizeRod():
    def VisualizeALL(self):
        plt.style.use("seaborn")
        fig_curve = plt.figure()
        curve = Axes3D(fig_curve)
        for num_frame in range(30, 55, 6):
            sim_nodes = np.load(
                "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                % num_frame)[:, :3]
            #curve.plot3D(sim_nodes[:, 0], sim_nodes[:, 1], sim_nodes[:, 2])

            x = sim_nodes[:, 0]
            y = sim_nodes[:, 1]
            z = sim_nodes[:, 2]

            curve.plot3D(x, y, z)
            curve.scatter(x, y, z)

        curve.set_xlabel('X')
        curve.set_ylabel('Y')
        curve.set_zlabel('Z')
        plt.show()
        #pdb.set_trace()

    def VisualizeIntermediaBackward(self):

        plt.style.use("seaborn")
        fig_curve = plt.figure()
        curve = Axes3D(fig_curve)

        for itr in range(0, 50, 15):
            iter_sim_nodes = np.load(
                "../DATA/CompareOverXPBDAndTomas/Tomas_Backward/" + str(itr) +
                "_29.npy")[:, :3]
            curve.plot3D(iter_sim_nodes[:, 0], iter_sim_nodes[:, 1],
                         iter_sim_nodes[:, 2])
            curve.scatter(iter_sim_nodes[:, 0], iter_sim_nodes[:, 1],
                          iter_sim_nodes[:, 2])

        curve.set_xlabel('X')
        curve.set_ylabel('Y')
        curve.set_zlabel('Z')

    def Visualize2DResult(self):

        ax = plt.gca()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.style.use("seaborn-notebook")
        for num_frame in range(30, 55, 5):
            sim_nodes = torch.as_tensor(
                np.load(
                    "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                    % num_frame))

            sim_nodes = sim_nodes[:, :3]
            cam_intrinsic_mat = torch.as_tensor(
                [[960.41357421875, 0.0, 1021.7171020507812],
                 [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])

            #plt.style.use("dark_background")
            pt_3d_project = torch.matmul(cam_intrinsic_mat,
                                         sim_nodes.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]
            X = sim_pbd[:, 0].detach().numpy()
            Y = sim_pbd[:, 1].detach().numpy()
            X_new = np.linspace(X.min(), X.max(), 30)

            spl = splrep(X, Y)
            y2 = splev(X_new, spl)

            plt.plot(X_new,
                     y2,
                     linewidth=5,
                     color="red",
                     linestyle="-",
                     alpha=0.1)
            #plt.scatter(X_new,-y2,s=30)
            plt.scatter(sim_pbd[:, 0], sim_pbd[:, 1], s=30, alpha=0.6)

            #ax_curve.plot3D(sim_nodes[:,0],sim_nodes[:,1],sim_nodes[:,2])
            #ax_curve.scatter(sim_nodes[:,0],sim_nodes[:,1],sim_nodes[:,2])
            #plt.axis("off")

            #plt.plot(x,-y,linewidth=1.0,color="white")

        RGB_IMAGE = plt.imread("../DATA/DATA_BaxterRope/rgb_raw_Baxter/30.png")
        plt.imshow(RGB_IMAGE)

        plt.show()

    def PlotPointOnSim2d(self):
        # ref:Shape Control of Deformable Linear Objects with Offline and Online Learning of Local Linear Deformation Models
        ax = plt.gca()

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        ax.set_xticks([])
        ax.set_yticks([])
        plt.style.use("seaborn-notebook")
        for num_frame in range(30, 52):
            fig, ax = plt.subplots()

            sim_nodes = torch.as_tensor(
                np.load(
                    "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                    % num_frame))

            sim_nodes = sim_nodes[:, :3]
            cam_intrinsic_mat = torch.as_tensor(
                [[960.41357421875, 0.0, 1021.7171020507812],
                 [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])

            #plt.style.use("dark_background")
            pt_3d_project = torch.matmul(cam_intrinsic_mat,
                                         sim_nodes.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]

            plt.scatter(sim_pbd[:, 0],
                        sim_pbd[:, 1],
                        s=20,
                        color="#FFB319",
                        alpha=1.0)
            # plt.plot(sim_pbd[:, 0],
            #             sim_pbd[:, 1],
                        
            #             alpha=1.0)

            RGB_IMAGE = plt.imread(
                "../DATA/DATA_BaxterRope/Blur_rgb_Baxter/%d.png" % num_frame)
            plt.imshow(RGB_IMAGE)
            plt.axis("off")

            fig.savefig("../DATA/PlotPointOnSim2d/%d.png" % num_frame)
        #plt.show()
        #pdb.set_trace()

    def PlotDifferenLossMatplotlib(self):
        fig = plt.figure()
        ax_curve = Axes3D(fig)
        
        
        index =1
       
        cal_methods = [ "projection","correspondance", "seg"]
        cal_tragets = [
             "centerline_3d_gravity", "centerline_2d" , "centerline_2d_plus_lwst"
        ]
        for cal_method in cal_methods:
            for cal_traget in cal_tragets:
                print(cal_method,cal_traget)
                plt.style.use('seaborn')
                #ax_curve = fig.add_subplot(3, 3, index, projection='3d')
                ax_curve.view_init(elev= -112., azim= -100)
                # ax_curve.set_xlabel("X")
                # ax_curve.set_ylabel("Y")
                # ax_curve.set_zlabel("Z")

                plt.locator_params(axis="z", nbins=6)
                #fig.subplots_adjust(left=0, right=0.6, bottom=0, top=1)
                
                index += 1
                ax_curve.title.set_text(str(cal_method)+' ' + str(cal_traget))
                for num_frame in [40]:
                    pcl = np.load("../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %num_frame)
                    sim_nodes = np.load("../DATA/save_loss_withConstConvergency/" +cal_method + "/" + cal_traget + "/frame_%d.npy" %num_frame)[:,:3]
                    ax_curve.scatter(sim_nodes[:,0],sim_nodes[:,1],sim_nodes[:,2])
                    ax_curve.plot3D(sim_nodes[:,0],sim_nodes[:,1],sim_nodes[:,2])
                    ax_curve.scatter(pcl[:,0],pcl[:,1],pcl[:,2])

                  
                    plt.savefig("./DATA/PlotDifferenLossMatplotlib/"+cal_method+"_"+cal_traget+".png")
                plt.cla()

        plt.show()
    

    def ConnectPlotDifferenLossMatplotlib(self):

        cal_methods = [ "projection","correspondance", "seg"]
        cal_tragets = [
             "centerline_3d_gravity", "centerline_2d" , "centerline_2d_plus_lwst"
        ]
        index = 0 
       
        Images =  np.zeros((550* 3, 800 * 3, 4))
        for cal_method in cal_methods:
            for cal_traget in cal_tragets:
                images = plt.imread("./DATA/PlotDifferenLossMatplotlib/"+cal_method+"_"+cal_traget+".png")
                x_index = int(index%3)
                y_index = int(index/3)
              
                Images[x_index*550 : (x_index+1) *550,y_index*800:(y_index+1)*800 ] = images
                index +=1
                #pdb.set_trace()
        
       
        
        plt.imshow(Images)
        plt.axis("off")
        plt.show()
    
    def ProjectionPtfrom3DNode(self, SimNode):
        #The cammer matrix, which can not be changed
        cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])

        pt_3d_project = torch.matmul(cam_intrinsic_mat, SimNode.float().T).T
        sim_pt_2d = torch.div(pt_3d_project, (pt_3d_project[:,
                                                            2]).reshape(-1, 1))
        SimNode_2D = sim_pt_2d[:, :2]
        return SimNode_2D
    
    def MergeImage(self):
        img1 = cv.imread("../DATA/DATA_BaxterRope/rgb_raw_Baxter/%d.png" % 40)
        img2 = cv.imread("../DATA/DATA_BaxterRope/rgb_raw_Baxter/%d.png" % 54)

        dst = cv.addWeighted(img1,0.3,img2,0.7,0)

        cal_methods = [ "projection","correspondance", "seg"]
        cal_tragets = [
             "centerline_3d_gravity", "centerline_2d" , "centerline_2d_plus_lwst"
        ]
        RGBIMAGE = cv.cvtColor(dst, cv.COLOR_BGR2RGB)
        for cal_method in cal_methods:
            for cal_traget in cal_tragets:

                for num_frame in [40,54]:
                    sim_nodes = torch.as_tensor(np.load("../DATA/save_loss_withConstConvergency/" +cal_method + "/" + cal_traget + "/frame_%d.npy" %num_frame)[:,:3])
                    SimNode_2D = self.ProjectionPtfrom3DNode(sim_nodes)
                    plt.scatter(SimNode_2D[:, 0], SimNode_2D[:, 1])
                    plt.plot(SimNode_2D[:, 0], SimNode_2D[:, 1])
                # RGBIMAGE = dst
                # RGBIMAGE[:,:,0] = dst[:,:,2]
                # RGBIMAGE[:,:,2] = dst[:,:,0]
                plt.imshow(RGBIMAGE)
                plt.axis("off")
                #plt.show()
                plt.savefig("../DATA/MergeImage/"+ cal_method+ "_" +cal_traget+ ".png" )
                plt.cla()
                #pdb.set_trace()
                
        # cv.imshow('dst',dst)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

    # def ConnectImage(self):
    #     cal_methods = [ "projection","correspondance", "seg"]
    #     #cal_methods = [ "projection"]
    #     cal_tragets = [
    #          "centerline_3d_gravity", "centerline_2d" , "centerline_2d_plus_lwst"
    #     ]
    #     images = []
    #     for cal_method in cal_methods:
    #         for cal_traget in cal_tragets:
    #             image = cv.imread("../DATA/MergeImage/"+ cal_method+ "_" +cal_traget+ ".png" )
    #             #image = cv.resize(image, (int(800),int(550)))
    #             #image=cv.cvtColor(image, cv.COLOR_BGR2RGB)
    #             images.append(image)
    #             # cv.imshow("test", image)
    #             # cv.waitKey(0) 
    #             # cv.destroyAllWindows() 

    #             # pdb.set_trace()
    #     #pdb.set_trace()
    #     images = np.array(images)
    #     # for i in range(1,4):
    #     #     plt.subplot(1,3,i)
    #     #     plt.imshow(images[i-1])
    #     #     plt.axis("off")
    #     # plt.show()
    #     #images = images.reshape(3,3,550,800,3)
    #     images = images.reshape(3,3,550,800,3)

    #     # def hconcat_resize(img_list, 
    #     #            interpolation 
    #     #            = cv.INTER_CUBIC):
    #     #     # take minimum hights
    #     #     h_min = min(img.shape[0] 
    #     #                 for img in img_list)
            
    #     #     # image resizing 
    #     #     im_list_resize = [cv.resize(img,
    #     #                     (int(img.shape[1] * h_min / img.shape[0]),
    #     #                         h_min), interpolation
    #     #                                 = interpolation) 
    #     #                     for img in img_list]
            
    #     #     # return final image
    #     #     #pdb.set_trace()
    #     #     return cv.hconcat(im_list_resize)
  
    #     # # function calling
    #     # im_tile = hconcat_resize(images)

    

    #     im_tile=cv.vconcat([cv.hconcat(im_list_h) for im_list_h in images])
    #     cv.imwrite("./DATA/ConnectImage_Tomas.png", im_tile)

    #     cv.imshow("test", im_tile)
  
    #     #waits for user to press any key 
    #     #(this is necessary to avoid Python kernel form crashing)
    #     cv.waitKey(0) 
        
    #     #closing all open windows 
    #     cv.destroyAllWindows() 
    
    def ConnectImage(self):
        num = 3
        index = 0

        IMAGES = np.zeros((1080* 3, 1920 * num, 3))
        frams = [30,54,60]
        #plt.subplot(2,1,1)
        for num_frame in frams:
            image = plt.imread("../DATA/PyElastica/PyElasticaRenderingImage/projection_centerline_3d_gravity/diag/frame_%04d.png" %int(num_frame-30))
            IMAGES[:1080, index*1920:(index+1)*1920  ] = image
            index += 1
        #plt.title("Tomas XPBD")
        plt.axis("off")
        #plt.imshow(IMAGES)


        index = 0
        #IMAGES = np.zeros((1080, 1920 * num, 3))
        for num_frame in frams:
            image = plt.imread("../DATA/PyElastica/PyElasticaRenderingImage/projection_centerline_2d/diag/frame_%04d.png" %int(num_frame-30))
            #pdb.set_trace()
            IMAGES[1080*1:1080* 2, index*1920:(index+1)*1920  ] = image
            index += 1

        #plt.subplot(2,1,2)
        index = 0
        #IMAGES = np.zeros((1080, 1920 * num, 3))
        for num_frame in frams:
            image = plt.imread("../DATA/PyElastica/PyElasticaRenderingImage/XPBD/projection_centerline_3d_gravity/diag/frame_%04d.png" %int(num_frame-30))
            #pdb.set_trace()
            IMAGES[1080*2:1080* 3, index*1920:(index+1)*1920  ] = image
            index += 1
        

      
        #plt.title("Tomas XPBD")
        plt.imshow(IMAGES)
        plt.legend()
        plt.axis("off")
        #plt.savefig("../DATA/ANALYSIS/ConnectImages.png",dpi=200)
        plt.show()