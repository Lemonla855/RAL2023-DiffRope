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


## ================================================================================
#   Init All the Data
## ================================================================================
class PBDRope():
    def __init__(self, start_frame, save_dir, LR_RATE, LAYERS, ITERATION,
                 Tomas_solve, MPCsteps, compliance):

        # --------- define number of particles ---------
        self.node_num = 30
        self.compliance = compliance
        self.dt = 0.1
        self.damper = 0.0

        # --------- Initialize Frame infomation ---------
        self.initial_frame = 0
        self.start_frame = start_frame  #54

        # --------- Judge save dir exits or not ---------
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --------- Initialize Backward Information ---------
        self.LR_RATE = LR_RATE
        # self.LAYERS = 10
        # self.STEPS = 50
        self.LAYERS = LAYERS
        self.MPCsteps = MPCsteps
        self.ITERATION = ITERATION

        # --------- Initialize Distance Constraint Type ---------
        self.Tomas_solve = Tomas_solve
        # --------- Actuated/Target ID ---------
        id_node_target = []
        id_node_actuated = []
        id_node_fixed = []

        for i in range(self.node_num):
            if i == 15:
                id_node_target.append(i)
            elif i == 17:
                id_node_actuated.append(i)

        # --------- Initialize ID ---------
        self.id_node_target = torch.as_tensor(id_node_target)
        self.id_node_actuated = torch.as_tensor(id_node_actuated)
        self.id_node_fixed = torch.as_tensor(id_node_fixed)

        # --------- Initialize Goal State ---------
        self.setGoalState()

        # --------- Initialize  ---------
        self.setInitialization()

        # --------- Weight for the Constraint ---------
        self.wab = 1.0
        self.wg = 1.0
        self.ws = 1.0
        self.wq = 1.0

        # ------- Constraints stiffness -------
        self.w_gravity = 0.1

        self.w_dis = 1
        self.w_strain = 1  ## effect : if on the same plane

        self.wq_strain = 1.0
        self.wq_bending = 1.0  ## effect : curvature

        self.w_SOR = 1.0

        # ------- Variable and Optimizer -------
        self.update_pos_actuated = torch.zeros((self.MPCsteps, 3)).float()
        for i in range(self.MPCsteps):
            self.update_pos_actuated[i] = self.node_pos_curr[
                self.id_node_actuated].clone().float()
        self.update_pos_actuated = Variable(self.update_pos_actuated,
                                            requires_grad=True)

        #pdb.set_trace()
        self.optimizer = optim.Adam([self.update_pos_actuated],
                                    lr=self.LR_RATE,
                                    weight_decay=0.000)
        torch.autograd.set_detect_anomaly(True)

    def setGoalState(self):

        # ------- Initialize Goal Position for the node -------
        self.node_pos_goal = torch.zeros((self.node_num, 3))
        for i in range(self.node_num):
            self.node_pos_goal[i] = torch.as_tensor([0, 2 * i, 0])

        # ------- Initialize Goal Radius for the node -------
        self.node_radius_goal = torch.ones(self.node_num)
        for i in range((self.node_num)):
            self.node_radius_goal[i] = 0.1

        # ------- Initialize Rest Lenght for the node -------
        self.goal_dist = self.RestLength(self.node_pos_goal)
        self.goal_dist[:] = 0.5

        # ------- Initialize Goal Volume1 for the node -------
        self.goal_volume1 = self.VolumeSim(self.node_pos_goal,
                                           self.node_radius_goal)

        # ------- Initialize Goal Volume2 for the node -------
        self.goal_volume2 = self.VolumeCom(self.node_pos_goal,
                                           self.node_radius_goal)

        # ------- Initialize Goal Quaternion for the node -------
        self.node_rotation, self.node_length_goal, self.node_quaternion_goal = self.Quaternion(
            self.node_pos_goal)

        # ------- Initialize Sim Shape for the node -------
        self.center, self.xp, self.ri = self.SimShape(self.node_pos_goal,
                                                      self.node_radius_goal)

        # ------- Initialize DarbouxRest for the node -------
        self.DarbouxVector = self.DarbouxRest(self.node_pos_goal,
                                              self.node_quaternion_goal)
        # self.node_pos_goal[15, 0] += 40

    def setInitialization(self):

        # ------- Initialize curr node position information for the node -------
        p_offset = torch.full((self.node_num, 3), 1.0)
        self.node_pos_curr = self.node_pos_goal * 1.0
        # self.node_pos_curr = self.node_pos_goal + \
        #     torch.as_tensor(p_offset)\
        #pdb.set_trace()
        self.node_pos_curr[self.id_node_actuated] = self.node_pos_goal[self.id_node_actuated] + \
             torch.as_tensor(p_offset[1])

        # ------- Initialize curr radius information for the node -------
        r_offset = torch.rand(self.node_num) * 0

        # ------- Initialize curr radius information for the node -------
        self.node_radius_curr = self.node_radius_goal + r_offset

        # ------- Initialize curr Quaternion information for the node -------
        _, _, self.node_quaternion_curr = self.Quaternion(self.node_pos_curr)

    # ------- Rest Length -------
    def RestLength(self, pos):
        diff_pos = torch.diff(pos, dim=0).float()
        diff_pos = torch.linalg.norm(diff_pos, dim=1)
        return diff_pos

    # ------- Simple Volume -------
    def VolumeSim(self, pos, radius):
        radius_a = radius[0:-1]
        radius_b = radius[1:]

        diff_pos = torch.diff(pos, dim=0).float()
        dis_pos = torch.linalg.norm(diff_pos, ord=None, dim=1)

        r0 = 0.5 * (radius_a + radius_b)

        vol = r0 * r0 * dis_pos
        return vol

    # ------- Complex Volume -------
    def VolumeCom(self, pos, radius):
        radius_a = radius[0:-1]
        radius_b = radius[1:]

        diff_pos = torch.diff(pos, dim=0).float()
        dist_pos = torch.linalg.norm(diff_pos, ord=None, dim=1)
        diff_radius = torch.diff(radius, dim=0)
        e = diff_radius / dist_pos

        L = dist_pos - diff_radius * e

        vol = math.pi / 3.0 * (
            (radius_a**3 - radius_b**3) * (e**3 - 3.0 * e) + L * (1.0 - e**2) *
            (radius_a**2 + radius_a * radius_b + radius_b**2))
        return vol

    # ------- Quaternion -------
    def Quaternion(self, pos):
        """
        Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        node_num = self.node_num
        source_vec = np.array([0, 0, 1])  #Source Vector
        q = torch.zeros((node_num - 1, 4))  #Quanternion mateix
        rest_length = torch.ones((node_num - 1, 1))
        rot = torch.zeros((node_num - 1, 3, 3))  #rotation matrix

        for i in range(pos.shape[0] - 1):

            pos_a, pos_b = pos[i, :3], pos[i + 1, :3]
            ab_vec = torch.as_tensor(
                pos_b -
                pos_a) * 1.0  # ab_vec is the vector between pt a and pt b
            ab_vec = (ab_vec / torch.linalg.norm(torch.FloatTensor(ab_vec)))
            ab_vec = ab_vec.detach().numpy()
            source_vec_normalized, ab_vec_normalized = (
                source_vec / np.linalg.norm(source_vec)).reshape(3), (
                    ab_vec / np.linalg.norm(ab_vec)).reshape(3)

            v = np.cross(source_vec_normalized, ab_vec_normalized)
            c = np.dot(source_vec_normalized, ab_vec_normalized)
            s = np.linalg.norm(v)

            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + \
                kmat.dot(kmat) * ((1 - c) / (s ** 2))
            r = R.from_matrix(rotation_matrix)
            rot[i] = torch.as_tensor(r.as_matrix())
            qc = r.as_quat()  # x,y,z,w'

            q[i] = torch.tensor(qc)
            rest_length[i] = torch.linalg.norm(
                torch.as_tensor(pos_b - pos_a) * 1.0)
        return rot, rest_length, q

    # ------- Simple Shape Matching -------
    def SimShape(self, pos, radius):
        n = self.node_num
        center = 0
        ri = 0
        xp = torch.zeros((n, 3))
        center = torch.sum(pos[:, :3], dim=0)
        center = center / (n * 1.0)
        xp = pos[:, :3] - center
        ri = torch.sum(radius, dim=0)
        ri /= n

        return center, xp, ri

    # ------- DarbouxRest -------
    def DarbouxRest(self, target, q):
        darbouxvector = torch.zeros((len(target) - 2, 4))

        for i in range(len(target) - 2):

            q0 = q[i, :]
            q1 = q[i + 1, :]
            omega = Quaternion(q0[3], q0[0], q0[1], q0[2]).conjugate * \
                Quaternion(q1[3], q1[0], q1[1], q1[2])
            w, x, y, z = omega.elements[0], omega.elements[1], omega.elements[
                2], omega.elements[3]
            darbouxRest = torch.as_tensor([x, y, z, w])
            darbouxRest = darbouxRest / np.linalg.norm(darbouxRest)
            # omega_plus = darbouxRest + np.array([0, 0, 0, 1])
            # omega_minus=darbouxRest - np.array([0, 0, 0, 1])
            # if np.linalg.norm(omega_minus,ord=2)>np.linalg.norm(omega_plus,ord=2):
            #		darbouxRest *= -1.0
            darbouxvector[i] = darbouxRest

        return darbouxvector

    # ------- mul of Quaternion  -------
    def mulQuaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p_scalar, q_scalar = p[:, 3], q[:, 3]
        p_imag, q_imag = p[:, :3], q[:, :3]
        quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
        quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
            -1, 1) * p_imag + torch.cross(p_imag, q_imag)
        return quat_imag, quat_scalar


## ================================================================================
#   Traj Shape
## ================================================================================
class TrajWaypoints():
    def HeartShape(self, node_step, offset):
        trajpoint_gt = torch.zeros((node_step, 3))
        for k in range(node_step):
            X = 12 / 99 * k
            trajpoint_gt[k, 0] = torch.as_tensor(16 * (np.sin(X))**3) * 1.0
            trajpoint_gt[k, 1] = torch.as_tensor(13 * np.cos(X) - 5 * np.cos(
                2 * X) - 2 * np.cos(3 * X) - np.cos(4 * X)) * 1.0 + offset

        print('\n', '============ Total number of heart waypoints : ',
              node_step, '============  \n')
        #pdb.set_trace()
        return trajpoint_gt

    def CubeShape(self, node_step, offset):

        trajpoint_gt = torch.zeros((node_step, 3))
        points = np.array([[0.0, 0.0, -0.01], [0, 0, -1], [-1, 0, -1],
                           [-1, -1, -1], [0, -1, -1], [0, 0, -1], [0, 0, 0]])

        num_pts = np.shape(points)[0]
        no_scale_final_pts = None

        for i in range(num_pts - 1):
            interp_pts = np.linspace(points[i],
                                     points[i + 1],
                                     num=int(node_step / (num_pts - 1)),
                                     endpoint=False)
            # interp_pts = np.linspace(points[i], points[i+1], num=int(8), endpoint=False)
            if i == 0:
                no_scale_final_pts = interp_pts
            else:
                no_scale_final_pts = np.vstack(
                    (no_scale_final_pts, interp_pts))

        no_scale_final_pts = np.vstack(
            (no_scale_final_pts, points[num_pts - 1, :]))

        # stretch = [[2.06498904e-01 , -6.30755443e-07 ,  1.07477548e-03],
        #  [1.61535574e-06 ,  1.18897198e-01 ,  7.85307721e-06],
        #  [7.08353661e-02 ,  4.48415767e-06 ,  2.05395893e-01]]
        stretch = np.eye(3)
        scale = 1
        num_final_pts = np.shape(no_scale_final_pts)[0]

        cube_points = np.zeros((num_final_pts, 3))
        for i in range(num_final_pts):
            cube_points[i, :] = np.dot(no_scale_final_pts[i, :], stretch)
        cube_points = scale * cube_points

        print('\n', '============ Total number of cube waypoints : ',
              num_final_pts, '============  \n')

        return cube_points

    def HorseShape(self, node_step, offset):
        trajpoint_gt = torch.zeros((node_step, 3))
        for k in range(node_step):
            theta = 2 * k * math.pi / 40
            desir_waypoint = np.array([
                -1 + 1 * np.sin(theta), -1 + 1 * np.cos(theta),
                -1 + 1 * np.sin(2 * theta)
            ])
            trajpoint_gt[k, 0] = desir_waypoint[0] * 10
            trajpoint_gt[k, 1] = desir_waypoint[1] * 10 + offset
            trajpoint_gt[k, 2] = desir_waypoint[2] * 10

        print('\n', '============ Total number of horse waypoints : ',
              node_step, '============  \n')
        return trajpoint_gt

    def PyramidShape(self, node_step, offset):

        points = np.array([[-0.01, -0.01, -0.01], [-1, -1, -1], [0, -2, 0],
                           [0, -2, -2], [-1, -1, -1], [0, 0, -2], [0, 0, 0]])

        num_pts = np.shape(points)[0]
        no_scale_final_pts = None

        for i in range(num_pts - 1):
            interp_pts = np.linspace(points[i],
                                     points[i + 1],
                                     num=int(node_step / (num_pts - 1)),
                                     endpoint=False)
            # interp_pts = np.linspace(points[i], points[i+1], num=int(8), endpoint=False)
            if i == 0:
                no_scale_final_pts = interp_pts
            else:
                no_scale_final_pts = np.vstack(
                    (no_scale_final_pts, interp_pts))

        no_scale_final_pts = np.vstack(
            (no_scale_final_pts, points[num_pts - 1, :]))

        # stretch = [[2.06498904e-01 , -6.30755443e-07 ,  1.07477548e-03],
        #  [1.61535574e-06 ,  1.18897198e-01 ,  7.85307721e-06],
        #  [7.08353661e-02 ,  4.48415767e-06 ,  2.05395893e-01]]
        stretch = np.eye(3)
        scale = 1
        num_final_pts = np.shape(no_scale_final_pts)[0]

        cube_points = np.zeros((num_final_pts, 3))
        for i in range(num_final_pts):
            cube_points[i, :] = np.dot(no_scale_final_pts[i, :], stretch)
        cube_points = scale * cube_points

        print('\n', '============ Total number of Pyramid waypoints : ',
              num_final_pts, '============  \n')

        return cube_points


## ================================================================================
#   Stretch Constraint
## ================================================================================


class AutoDiffStretch():
    def __init__(self, compliance, dt, rest_length):
        self.compliance = compliance
        self.dt = dt
        self.rest_length = rest_length

    def C_strain_solve(self, pos, Lamda, quat, wa, wb, wq, itr):

        dq = torch.zeros((len(pos) - 1, 4))
        dx = torch.zeros((len(pos), 3))

        dx = torch.zeros((len(pos), 3))
        a = pos[:-1, :3]
        b = pos[1:, :3]
        d3 = torch.zeros((pos.shape[0] - 1, 3))
        qc = quat[:]

        # f = qc[:, 3]
        # x = qc[:, 0]
        # y = qc[:, 1]
        # z = qc[:, 2]
        # qc[0] = torch.as_tensor([0.719036, -0.296816, 0.000071, 0.628402])

        # d3[:, 0] = 2.0 * (qc[:, 0] * qc[:, 2] + qc[:, 3] * qc[:, 1])
        # d3[:, 0] = 2.0 * (torch.add(qc[:, 0] * qc[:, 2], qc[:, 3] * qc[:, 1]))
        # d3[:, 1] = 2.0 * (qc[:, 1] * qc[:, 2] - qc[:, 3] * qc[:, 0])

        d3[:, 0] = 2.0 * (torch.add(qc[:, 0] * qc[:, 2], qc[:, 3] * qc[:, 1]))
        d3[:, 1] = 2.0 * torch.sub(qc[:, 1] * qc[:, 2], qc[:, 3] * qc[:, 0])
        d3[:, 2] = qc[:, 3] * qc[:, 3] - qc[:, 0] * \
            qc[:, 0] - qc[:, 1] * qc[:, 1] + qc[:, 2] * qc[:, 2]

        Cx = torch.div(b - a, self.rest_length.reshape(-1, 1)) - d3
        #pdb.set_trace()

        alpha = self.compliance / (self.dt * self.dt)

        l_sqaure = self.rest_length**2

        # La[0] = torch.as_tensor([-0.000138, -0.000334, -0.000055])

        dL = torch.div(
            (-Cx - alpha * Lamda) * l_sqaure.reshape(-1, 1),
            (wa + wb + 4 * l_sqaure * wq + alpha * l_sqaure).reshape(-1, 1))

        dq_v = -2 * wq * dL

        quat_1 = torch.zeros((dq_v.shape[0], 4))
        quat_2 = torch.zeros((dq_v.shape[0], 4))
        quat_1[:,
               0], quat_1[:,
                          1], quat_1[:,
                                     2], quat_1[:,
                                                3] = dq_v[:,
                                                          0], dq_v[:,
                                                                   1], dq_v[:,
                                                                            2], 0
        quat_2[:, 0], quat_2[:, 1], quat_2[:, 2], quat_2[:, 3] = - \
            qc[:, 1], qc[:, 0], -qc[:, 3], qc[:, 2]

        dq_imag, dq_scalar = self.mul_quaternion(quat_1,
                                                 quat_2)  # p,q is x,y,z,w

        # ----------  UPDATES : Eq. 37  ----------
        dp = torch.div(dL, self.rest_length.reshape(-1, 1))

        dx[:-1] += -wa * dp
        dx[1:] += +wb * dp
        dq = torch.hstack((dq_imag, dq_scalar.reshape(-1, 1)))  # x,y,z,w

        Lamda += dL
        # print(dp)
        #if itr == 1:
        #    pdb.set_trace()

        return dx, Lamda, dq, Cx

    def mul_quaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p_scalar, q_scalar = p[:, 3], q[:, 3]
        p_imag, q_imag = p[:, :3], q[:, :3]
        quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
        quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
            -1, 1) * p_imag + torch.cross(p_imag, q_imag)
        # omega = torch.hstack((quat_imag, quat_scalar.reshape(-1, 1)))

        return quat_imag, quat_scalar


## ================================================================================
#   Distance Constraint
## ================================================================================


class AutoDiffDistance():
    def __init__(self, rest_length, compliance, dt):
        self.rest_length = rest_length
        self.compliance = compliance
        self.dt = dt

    def solveDistConstraint(self, pos, Lamda, wa, wb):

        dx = torch.zeros((len(pos), 3))
        diff_pos = torch.diff(pos, dim=0).float()
        diff_pos_norm = torch.linalg.norm(diff_pos, dim=1)
        diff_pos_transpose = diff_pos.T

        diff_pos_normalized = torch.div(-diff_pos_transpose, diff_pos_norm).T
        alpha = self.compliance / (self.dt * self.dt)
        C_dist = diff_pos_norm - self.rest_length[:]
        # print(c)
        # pdb.set_trace()

        dL = torch.sub(-alpha * Lamda.T, C_dist)
        dL = dL.T / (wa + wb + alpha)
        delta_x = torch.mul(dL, diff_pos_normalized)
        Lamda += dL
        dx[:-1] += delta_x * wa
        dx[1:] -= delta_x * wb
        # pdb.set_trace()
        return dx, Lamda, C_dist

    def solveDistConstraintTDMALinearSolver(self, pos, Lamda, wa, wb):

        ##  K : number of constraints
        ##  N : number of nodes

        weight = wa
        vec_weight = torch.zeros(3 * len(pos)) + weight
        vec_weight[0:3] = 0
        vec_weight[-3:] = 0
        mat_weight = torch.diag(vec_weight, diagonal=0)

        dx = torch.zeros((len(pos), 3))
        diff_pos = torch.diff(pos, dim=0).float()
        diff_pos_norm = torch.linalg.norm(diff_pos, dim=1)
        diff_pos_transpose = diff_pos.T
        constr_dis = diff_pos_norm - self.rest_length[:]

        gradC_p = torch.div(
            -diff_pos_transpose,
            diff_pos_norm)  ## dim : 3*(N-1) : should be p_{i} - p_{i+1}
        gradC_p_T = gradC_p.T  ## dim : (N-1)*3
        # gradC_pp1_T = -gradC_p_T

        num_const = gradC_p.T.shape[0]
        mat_gradC_p = torch.zeros(
            (num_const, 3 * (num_const + 1)))  ## dim : K*(N*3)

        id_diag0_col = torch.arange(num_const * 3).reshape(
            -1, 3)  ## diag0 ID : col
        id_diag0_row = torch.arange(num_const).reshape(-1, 1).repeat(
            1, 3)  ## diag0 ID : row
        mat_gradC_p[id_diag0_row, id_diag0_col] = gradC_p.T

        id_diagp1_col = torch.arange(3, (num_const + 1) * 3, 1).reshape(-1, 3)
        id_diagp1_row = id_diag0_row
        mat_gradC_p[id_diagp1_row, id_diagp1_col] = -gradC_p.T

        gradCp_W_CpT = torch.matmul(torch.matmul(mat_gradC_p, mat_weight),
                                    mat_gradC_p.T)

        # ## https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        # ## vec_a & vec_c should have the same length as vec_b
        # ## then : vec_a[0] = 0 / vec_c[-1] = 0
        vect_b = torch.diag(gradCp_W_CpT, diagonal=0)

        vect_a = torch.diag(gradCp_W_CpT, diagonal=-1)
        vect_a = torch.hstack((torch.zeros(1), vect_a))

        vect_c = torch.diag(gradCp_W_CpT, diagonal=1)
        vect_c = torch.hstack((vect_c, torch.zeros(1)))

        vect_d = -constr_dis

        ### Using TDMA to solve the equation
        Lamda = self.solveTDMA(vect_a, vect_b, vect_c, vect_d)

        dx = torch.matmul(torch.matmul(mat_weight, mat_gradC_p.T),
                          Lamda).reshape(-1, 3)

        # pdb.set_trace()

        # gradCp_CpT = torch.diagonal(torch.matmul(gradC_p_T[1:, :], gradC_p[:, :-1]))  ## dim : (N-1)*1
        # vec_b = torch.zeros_like(diff_pos_norm) - 2 * weight
        # # vec_b[0] = -2
        # # vec_b[-1] = -2
        # vec_b[0] = -1
        # vec_b[-1] = -1

        # vec_a = torch.hstack((torch.zeros(1), gradCp_CpT))
        # vec_c = torch.hstack((grtorch.matmul(torch.matmul(mat_weight, mat_gradC_p.T), Lamda)adCp_CpT, torch.zeros(1)))
        # vec_d = const_dis

        # Lamda = self.solveTDMA(vec_a, vec_b, vec_c, vec_d)

        # thomas_A_mat = self.getTridiagMatrix(vec_a, vec_b, vec_c, -1, 0, 1)
        # thomas_b_vec = const_dis

        # # dx = torch.matmul(gradC_p, Lamda)
        # gradC_p_x = torch.vstack((torch.diag(gradC_p[0, :]), torch.zeros_like(gradC_p[0, :])))  # x
        # gradC_p_y = torch.vstack((torch.diag(gradC_p[1, :]), torch.zeros_like(gradC_p[0, :])))  # x
        # gradC_p_z = torch.vstack((torch.diag(gradC_p[2, :]), torch.zeros_like(gradC_p[0, :])))  # x
        # gradC_p_x[-1, -1] = -gradC_p[0, -1]
        # gradC_p_y[-1, -1] = -gradC_p[1, -1]
        # gradC_p_z[-1, -1] = -gradC_p[2, -1]

        # dx_x = torch.matmul(gradC_p_x, Lamda)
        # dx_y = torch.matmul(gradC_p_y, Lamda)
        # dx_z = torch.matmul(gradC_p_z, Lamda)

        # dx = torch.vstack((dx_x, dx_y, dx_z)).T

        # num_const = gradC_p.T.shape[0]
        # mat_gradC_p = torch.zeros((3 * (num_const + 1), num_const))
        # id_diag0 = torch.arange(num_const * 3).reshape(-1, 3)
        # id_diagN1 = torch.arange(3, (num_const + 1) * 3, 1).reshape(-1, 3)
        # mat_gradC_p[id_diag0, torch.arange(num_const).reshape(-1, 1).repeat(1, 3)] = gradC_p.T
        # mat_gradC_p[id_diagN1, torch.arange(num_const).reshape(-1, 1).repeat(1, 3)] = -gradC_p.T

        # mat_gradC_p[0:3, 0] = 0
        # mat_gradC_p[-3:, -1] = 0

        # dx = torch.matmul(mat_gradC_p, Lamda).reshape(-1, 3)

        #pdb.set_trace()

        return dx, Lamda, constr_dis

    # def getTridiagMatrix(self, a, b, c, k1=-1, k2=0, k3=1):
    #     return torch.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

    def solveTDMA(self, vec_a, vec_b, vec_c, vec_d):
        '''
        TDMA solver, a b c d can be NumPy array type or Python list type.
        refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
        http://www.thevisualroom.com/tri_diagonal_matrix.html#id10
        '''
        length = vec_a.shape[0]

        vec_b_list = [vec_b[0]]
        vec_d_list = [vec_d[0]]
        for it in range(1, length):
            # mc = vec_a[it] / vec_b[it - 1]
            # vec_b[it] = vec_b[it] - mc * vec_c[it - 1]
            # vec_d[it] = vec_d[it] - mc * vec_d[it - 1]
            mc = vec_a[it] / vec_b_list[-1]
            vec_b_temp = vec_b[it] - mc * vec_c[it - 1]
            vec_d_temp = vec_d[it] - mc * vec_d_list[-1]
            vec_b_list.append(vec_b_temp)
            vec_d_list.append(vec_d_temp)

        vec_b = torch.stack(vec_b_list)
        vec_d = torch.stack(vec_d_list)

        xc_list = []
        xc_list.append(vec_d[-1] / vec_b[-1])
        for il in range(length - 2, -1, -1):
            xc_temp = (vec_d[il] - vec_c[il] * xc_list[-1]) / vec_b[il]
            xc_list.append(xc_temp)

        xc = torch.stack(xc_list)
        xc = torch.flip(xc, dims=[0])

        #pdb.set_trace()

        return xc

    def getConvergencyDistanceConstraint(self, C_dis):

        C_dis_max = torch.max(torch.abs(C_dis))

        return C_dis_max


## ================================================================================
#   SimVolume Constraint
## ================================================================================


class AutoDiffSimVolume():
    def __init__(self, compliance, dt, Vi):

        self.compliance = compliance
        self.dt = dt
        self.Vi = Vi

    def solveVolumeConstraint(self, pos, radius, lamda):

        dx = torch.zeros((pos.shape[0], 3))
        dr = torch.zeros((pos.shape[0]))
        ra = radius[:-1]
        rb = radius[1:]

        s = 0.5 * (ra + rb)
        ab = torch.diff(pos, dim=0).float()
        R3 = torch.div(ab.T, torch.linalg.norm(ab, dim=1)).T
        r3ab = torch.sum(R3 * (ab), dim=1)
        C_simvol = torch.mul(r3ab.T, s * s).T - self.Vi
        W = C_simvol
        gp = torch.mul(R3.T, -s * s).T

        gr = s * r3ab
        gr = s * r3ab
        wSum = (1 + 1) * gr * gr + (1 + 1) * \
            torch.linalg.norm(gp, 2, dim=1) + 1e-6
        lamda = W / wSum

        dp = torch.mul(gp.T, -lamda).T

        drr = -lamda * gr
        w = 1.0
        dr[:-1] += drr
        dr[1:] += drr
        dx[:-1] += dp * w
        dx[1:] += -dp * w

        return dx, dr, lamda, C_simvol


## ================================================================================
#   Complex Volume Constraint
## ================================================================================


class AutoDiffComVolume():
    def __init__(self, compliance, dt, volume):
        self.compliance = compliance
        self.dt = dt
        self.volume = volume

    def solveVolumeConstraint(self, pos, radius, Lambda):
        dx = torch.zeros((pos.shape[0], 3))
        dr = torch.zeros((pos.shape[0]))
        #Lambda = torch.zeros(pos.shape[0] - 1)
        # update_pos=Variable(copy.deepcopy(pos), requires_grad=True)
        # update_pos = pos

        diff_radius = torch.diff(radius, dim=0).float()
        diff_pos = -torch.diff(pos, dim=0).float()
        fff = torch.linalg.norm(diff_pos, dim=1)
        e = diff_radius / torch.linalg.norm(diff_pos, dim=1)

        L = torch.linalg.norm(diff_pos, dim=1) + (-diff_radius) * e
        e2 = e * e
        e3 = e2 * e
        e4 = e3 * e
        ra = torch.clone(radius[:-1])
        rb = torch.clone(radius[:-1])

        ra2 = torch.matmul(ra, ra)

        rb2 = torch.matmul(rb, rb)
        ra3 = ra**3
        rb3 = rb**3

        rarb = ra**2 + ra * rb + rb**2

        gpa_n = torch.div(
            (math.pi * (e - e3) * (ra3 - rb3) / fff + math.pi / 3.0 * rarb *
             (1.0 - 3.0 * e4 + 2.0 * e2)).T, self.volume).T

        gra = torch.div(
            (math.pi * (ra2 * (1.0 - e2) / fff + ra2 * e3 - ra2 * 3.0 * e) +
             math.pi * (e2 - 1.0) * rb3 / fff + math.pi / 3.0 *
             (+4.0 * rarb * (e - e3) + L * (1.0 - e2) * (2.0 * ra + rb))).T,
            self.volume).T
        grb = torch.div(
            (math.pi * (rb3 / fff * (1.0 - e2) + 3.0 * rb2 * e - e3 * rb2) +
             math.pi * ra3 / fff * (e2 - 1.0) + math.pi / 3.0 *
             (-4.0 * rarb * (e - e3) + L * (1.0 - e2) * (ra + 2.0 * rb))).T,
            self.volume)
        ban = torch.div(diff_pos.T, fff).T

        wSum = 0
        wSum = wSum + (1 + 1) * gpa_n * gpa_n + \
            1 * gra * gra + 1 * grb * grb

        V = math.pi / 3.0 * \
            ((ra3 - rb3) * (e3 - 3.0 * e) + L * (1.0 - e2) * rarb)
        #pdb.set_trace()

        C_compvol = torch.div(V, self.volume) - 1.0

        alpha = self.compliance / (self.dt * self.dt)
        dL = torch.sub(-alpha * Lambda, C_compvol)

        dL = dL / (wSum + alpha)

        dpa = torch.mul(ban.T, (gpa_n * dL)).T
        # print(dpa)
        dra = gra * dL
        drb = grb * dL

        dx[:-1] += dpa
        dx[1:] -= dpa

        dr[:-1] += dra
        dr[1:] += drb

        return dx, dr, Lambda, C_compvol


## ================================================================================
#   Shape Matching Constraint
## ================================================================================


class AutoSimShape():
    def __init__(self, xp, ri, n):
        self.xp = xp
        self.ri = ri
        self.n = n

    def solveShapeMatchingConstraint(self, pos, radius, constraint_q, L):
        x = pos.detach()
        center = torch.zeros((3))
        A = torch.zeros((3, 3))
        center = torch.sum(x[:, :3], dim=0)
        center /= self.n
        x[:, :3] = x[:, :3] - center
        A = torch.mm(x[:, :3].T, self.xp[:, :3])
        for i in range(3):

            qua = constraint_q.elements
            # print(qua)
            rot = R.from_quat([qua[1], qua[2], qua[3], qua[0]])

            rotation = torch.FloatTensor(rot.as_matrix())

            omega = (torch.cross(rotation[:, 0], A[:, 0]) +
                     torch.cross(rotation[:, 1], A[:, 1]) +
                     torch.cross(rotation[:, 2], A[:, 2])) * (
                         1.0 / abs(rotation[:, 0].dot(A[:, 0]) +
                                   rotation[:, 1].dot(A[:, 1]) +
                                   rotation[:, 2].dot(A[:, 2])) + 1.0e-9)
            w = torch.linalg.norm(omega)

            if w < 1e-6:
                break

            constraint_q = Quaternion(axis=(1 / w) * omega,
                                      angle=w) * constraint_q
            constraint_q = constraint_q.normalised

        scale = torch.sum(radius, dim=0)
        scale /= self.n * self.ri
        wSum = 0
        grad = torch.zeros((self.n, 3))

        cc = constraint_q.elements
        ro = R.from_quat([cc[1], cc[2], cc[3], cc[0]])
        roo = torch.FloatTensor(rot.as_matrix())

        previous_x = self.xp
        kk = x - torch.mm(roo, (previous_x.T) * scale).T
        grad = kk
        wSum = wSum + torch.sum(torch.linalg.norm(kk, 2, dim=1))

        alpha = 0.2

        dL = (-wSum - alpha * L) / (wSum + alpha)

        dx = torch.zeros((self.n, 3))
        corr = dL * grad
        dx = corr * 1.0

        L += dL
        return dx, constraint_q, L


## ================================================================================
#   Complex Shape Matching Constraint
## ================================================================================


class AutoComShape():
    def __init__(self, xi, ri, qi, n):

        self.xi = xi
        self.qi = qi
        self.ri = ri
        self.n = n

    def solveShapeMatchingConstraint(self, x, q, r, constraint_q):
        n = self.n
        qi = self.qi
        xi = self.xi
        ri = self.ri
        T_target = torch.ones((n, 3, 4))
        T_source = torch.ones((n, 3, 4))
        for i in range(n):

            q0 = Quaternion(q[i, 3], q[i, 0], q[i, 1], q[i, 2])
            q1 = Quaternion(q[i + 1, 3], q[i + 1, 0], q[i + 1, 1], q[i + 1, 2])
            temp = r[i] * Quaternion.slerp(q0, q1, 0.5)

            T_target[i, :3, :3] = torch.FloatTensor(temp.rotation_matrix)

            T_target[i, :3, 3] = x[i + 1]
        for i in range(n):
            q0 = Quaternion(qi[i, 3], qi[i, 0], qi[i, 1], qi[i, 2])
            q1 = Quaternion(qi[i + 1, 3], qi[i + 1, 0], qi[i + 1, 1], qi[i + 1,
                                                                         2])
            tt = ri[i] * Quaternion.slerp(q0, q1, 0.5)
            T_source[i, :3, :3] = torch.FloatTensor(tt.rotation_matrix)
            T_source[i, :3, 3] = xi[i + 1]
        mean_source = torch.zeros(3)
        mean_target = torch.zeros(3)
        for i in range(n):
            mean_source += x[i + 1]
            mean_target += xi[i + 1]

        mean_source = torch.FloatTensor(mean_source) / n
        mean_target = torch.FloatTensor(mean_target) / n
        for i in range(n):

            T_target[i, :, 3] = T_target[i, :, 3] - mean_target
            T_source[i, :, 3] = T_source[i, :, 3] - mean_source
        A = torch.zeros((3, 3))
        for i in range(n):

            A += torch.mm(T_target[i, :, :], T_source[i, :, :].T)
        A = A / (4.0 * n)

        for i in range(3):

            qua = constraint_q.elements
            # print(qua)
            rot = R.from_quat([qua[1], qua[2], qua[3], qua[0]])

            rotation = torch.FloatTensor(rot.as_matrix())

            omega = (torch.cross(rotation[:, 0], A[:, 0]) +
                     torch.cross(rotation[:, 1], A[:, 1]) +
                     torch.cross(rotation[:, 2], A[:, 2])) * (
                         1.0 / abs(rotation[:, 0].dot(A[:, 0]) +
                                   rotation[:, 1].dot(A[:, 1]) +
                                   rotation[:, 2].dot(A[:, 2])) + 1.0e-9)
            w = torch.linalg.norm(omega)

            if w < 1e-6:
                break

            constraint_q = Quaternion(axis=(1 / w) * omega,
                                      angle=w) * constraint_q
            constraint_q = constraint_q.normalised
        qua = constraint_q.elements
        # print(qua)
        rot = R.from_quat([qua[1], qua[2], qua[3], qua[0]])

        rotation = torch.FloatTensor(rot.as_matrix())
        nom = 0.0
        den = 0.0
        for i in range(n):
            a = T_source[i, :, :]
            b = T_target[i, :, :]
            nom += torch.sum(torch.mm(rotation, a) * (b))

            den += torch.sum(a * a)

        s = 1.0 * nom / den

        Rm = torch.matmul(rotation, mean_source)

        t = mean_target - s * Rm

        qR = Quaternion(matrix=rot.as_matrix())

        dx = torch.zeros((BaxterRope.node_num, 3))
        dr = torch.zeros(BaxterRope.node_num)
        dq = torch.zeros((BaxterRope.node_num, 4))
        for i in range(-1, n):
            ddx = [0, 0, 0]
            ddx = s * torch.matmul(
                rotation, (xi[i + 1] - mean_source)) + mean_target - x[i + 1]

            da = (constraint_q *
                  Quaternion(qi[i, 3], qi[i, 0], qi[i, 1], qi[i, 2]))

            daaq = da.elements
            dqa = torch.FloatTensor([daaq[1], daaq[2], daaq[3], daaq[0]
                                     ]) - q[i]
            db = (constraint_q * Quaternion(qi[i + 1, 3], qi[i + 1, 0],
                                            qi[i + 1, 1], qi[i + 1, 2]))
            dbbq = db.elements
            dqb = torch.FloatTensor([dbbq[1], dbbq[2], dbbq[3], dbbq[0]
                                     ]) - q[i + 1]

            drr = s * ri[i + 1] - r[i + 1]
            dx[i + 1] += ddx
            dr[i + 1] += drr
            dq[i] += dqa
            dq[i + 1] += dqb
        return dx, dr, dq, constraint_q


## ================================================================================
#  Raius Constraint
## ================================================================================


class AutoDiffRadius():
    def __init__(self, compliance, dt, r):
        self.compliance = compliance
        self.dt = dt
        self.r = r

    def C_radius_solve(self, L, r_cur):
        dr = torch.zeros((len(self.r)))
        alpha = self.compliance / (self.dt * self.dt)
        C_radius = torch.div(r_cur, self.r) - 1.0
        dL = (-C_radius - alpha * L) / (1 / (self.r * self.r) + alpha)
        drr = 1 / self.r * dL
        dr = drr
        L += dL

        return dr, L, C_radius


## ================================================================================
#   Bending Constraint
## ================================================================================


class AutoDiffBending():
    def __init__(self, compliance, dt, darbouxRest):
        self.compliance = compliance
        self.dt = dt
        self.darbouxRest = darbouxRest

    def solve_BendTwistConstraint(self, target, quat, Lambda, wq, wu):

        # x,y,z,w

        q0 = quat[:-1]
        q1 = quat[1:, :]

        # q0[0] = torch.as_tensor([0.682700, -0.318297, -0.000019, 0.657730])
        # q1[0] = torch.as_tensor([0.804539, -0.374640, -0.000135, 0.460827])
        P = torch.zeros_like(quat[:-1])
        P[:, 3] = quat[:-1, 3]
        P[:, :3] = -1 * (quat[:-1, :3])
        Q = quat[1:, :]

        # Qa = torch.as_tensor(
        #    [-0.018268, 0.217283, -0.182538, 0.958715]).reshape(-1, 4)
        # Qb = torch.as_tensor(
        #    [0.365269, -0.782895, -0.027047, 0.502914]).reshape(-1, 4)

        omega_PQ, omega_pq0 = self.mul_quaternion(P, Q)
        omega = torch.hstack((omega_PQ, omega_pq0.reshape(-1, 1)))
        #omega = torch.div(omega, torch.linalg.norm(omega, dim=1))

        # self.darbouxRest[0] = torch.as_tensor(
        #    [0.214160, -0.099520, -0.000316, 0.971716])

        delta_omega = torch.zeros_like(omega)

        #darboux_method_A = self.getDarbouxVect(quat)
        #pdb.set_trace()

        omega_plus = omega + self.darbouxRest
        omega_minus = omega - self.darbouxRest
        for i in range(omega_minus.shape[0]):
            if torch.linalg.norm(omega_minus[i], ord=2) > torch.linalg.norm(
                    omega_plus[i], ord=2):
                delta_omega[i] = omega_plus[i]
            else:
                delta_omega[i] = omega_minus[i]

        # alpha = 0.0
        alpha = self.compliance / (self.dt * self.dt)
        Cx = torch.as_tensor(delta_omega[:, :3])
        # print(lamda[i,:].shape)

        dL = -(Cx + alpha * Lambda) / (wq + wu + alpha)
        # dL/=np.linalg.norm(dL)
        dLq = torch.zeros((dL.shape[0], 4))
        dLq[:, 0], dLq[:, 1], dLq[:, 2] = dL[:, 0], dL[:, 1], dL[:, 2]

        pq_image, pq_scalar = self.mul_quaternion(q1, dLq)  # x,y,z,w
        da = torch.hstack((pq_image, pq_scalar.reshape(-1, 1)))

        pq_image, pq_scalar = self.mul_quaternion(q0, dLq)  # x,y,z,w
        db = torch.hstack((pq_image, pq_scalar.reshape(-1, 1)))

        dq = torch.zeros((target.shape[0] - 1, 4))
        dq[:-1] += -da * wq
        dq[1:] += db * wu
        Lambda += dL
        #pdb.set_trace()

        return dq, Lambda, Cx

    def mul_quaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p_scalar, q_scalar = p[:, 3], q[:, 3]
        p_imag, q_imag = p[:, :3], q[:, :3]
        quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
        quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
            -1, 1) * p_imag + torch.cross(p_imag, q_imag)
        # omega = torch.hstack((quat_imag, quat_scalar.reshape(-1, 1)))

        return quat_imag, quat_scalar


## ================================================================================
#   Process the optimization Data
## ================================================================================


class ProcessOptData():
    # -------- Show the Image --------
    def ShowImage(self, ITERATION, trajwaypoints, sim_node, save_loss_itr,
                  save_C_dis_max_itr, save_C_bending_max_itr,
                  save_C_strain_max_itr):
        print("==========================================================")

        print("\n")
        fig = plt.figure(figsize=(15, 8))
        ax_curve = Axes3D(fig)

        # -------- set label and axis limitation --------
        #ax_curve.set_xlim3d([-2, 2])
        ax_curve.set_xlabel('X')
        #ax_curve.set_ylim3d([0, 40])
        ax_curve.set_ylabel('Y')
        ax_curve.set_zlim3d([-2, 10])
        ax_curve.set_zlabel('Z')

        # -------- Plot simpoint  --------
        ax_curve.plot3D(sim_node[:, 0].detach(), sim_node[:, 1].detach(),
                        sim_node[:, 2].detach())
        ax_curve.scatter(sim_node[:, 0].detach(), sim_node[:, 1].detach(),
                         sim_node[:, 2].detach())
        ax_curve.plot3D(trajwaypoints[:, 0], trajwaypoints[:, 1],
                        trajwaypoints[:, 2])
        ax_curve.scatter(trajwaypoints[:, 0], trajwaypoints[:, 1],
                         trajwaypoints[:, 2])
        # ax = axes.ravel()
        # ax_3d_curve = fig.add_subplot(2, 3, 1, projection='3d')
        # # init_plot_data = np.array([0, 0, 0])
        # line1_real_3d, = ax_3d_curve.plot([], [],
        #                                   color='#8bc24c',
        #                                   linestyle='-',
        #                                   linewidth=2)
        # line2_sim_3d, = ax_3d_curve.plot([], [],
        #                                  color='#de4307',
        #                                  linestyle='--',
        #                                  linewidth=1)
        # scatter1_sim_3d = ax_3d_curve.scatter([], [], [],
        #                                       s=16,
        #                                       c='#8134af',
        #                                       alpha=0.8)
        # # line3, = ax.plot(0, 0, color='#1F6ED4', marker='o', markersize=6)
        # ax_3d_curve.set_xlim3d([-10, 10])
        # ax_3d_curve.set_xlabel('X')
        # ax_3d_curve.set_ylim3d([0, 40])
        # ax_3d_curve.set_ylabel('Y')
        # ax_3d_curve.set_zlim3d([-5, 5])
        # ax_3d_curve.set_zlabel('Z')

        # ax_loss = fig.add_subplot(2, 3, 2)
        # line3_loss, = ax_loss.plot([], [],
        #                            color='#1F6ED4',
        #                            marker='o',
        #                            markersize=3,
        #                            linestyle='-',
        #                            linewidth=1)
        # ax_loss.set_xlim(0, ITERATION)
        # ax_loss.set_ylim(0, 100)
        # ax_loss.set_xlabel('iter (steps)')  # Add an x-label to the axes.
        # ax_loss.set_ylabel('loss')
        # ax_loss.set_title('loss over frames')

        # ax_itr_C_dis_max = fig.add_subplot(2, 3, 4)
        # line4_itr_C_dis_max, = ax_itr_C_dis_max.plot([], [],
        #                                              color='#1F6ED4',
        #                                              marker='o',
        #                                              markersize=3,
        #                                              linestyle='-',
        #                                              linewidth=1)
        # ax_itr_C_dis_max.set_xlim(0, ITERATION)
        # ax_itr_C_dis_max.set_ylim(0, 0.1)
        # ax_itr_C_dis_max.set_xlabel(
        #     'iter (steps)')  # Add an x-label to the axes.
        # ax_itr_C_dis_max.set_ylabel('max C_dis')
        # ax_itr_C_dis_max.set_title('Max C_dis constraint convergency')

        # ax_itr_C_bending_max = fig.add_subplot(2, 3, 5)
        # line5_itr_C_bending_max, = ax_itr_C_bending_max.plot([], [],
        #                                                      color='#1F6ED4',
        #                                                      marker='o',
        #                                                      markersize=3,
        #                                                      linestyle='-',
        #                                                      linewidth=1)
        # ax_itr_C_bending_max.set_xlim(0, ITERATION)
        # ax_itr_C_bending_max.set_ylim(0, 0.1)
        # ax_itr_C_bending_max.set_xlabel(
        #     'iter (steps)')  # Add an x-label to the axes.
        # ax_itr_C_bending_max.set_ylabel('max C_bending')
        # ax_itr_C_bending_max.set_title('Max C_bending constraint convergency')

        # ax_itr_C_strain_max = fig.add_subplot(2, 3, 6)
        # line6_itr_C_strain_max, = ax_itr_C_strain_max.plot([], [],
        #                                                    color='#1F6ED4',
        #                                                    marker='o',
        #                                                    markersize=3,
        #                                                    linestyle='-',
        #                                                    linewidth=1)
        # ax_itr_C_strain_max.set_xlim(0, ITERATION)
        # ax_itr_C_strain_max.set_ylim(0, 2)
        # ax_itr_C_strain_max.set_xlabel(
        #     'iter (steps)')  # Add an x-label to the axes.
        # ax_itr_C_strain_max.set_ylabel('max C_strain')
        # ax_itr_C_strain_max.set_title('Max C_strain constraint convergency')

        # #def ShowImage(self, centerline_3d_gravity, sim_node):
        # line1_real_3d_data = trajwaypoints.detach().numpy()
        # line1_real_3d.set_data(line1_real_3d_data[:, 0], line1_real_3d_data[:,
        #                                                                     1])
        # line1_real_3d.set_3d_properties(line1_real_3d_data[:, 2])

        # # scatter1_sim_3d._offsets3d = (line1_real_3d_data[:, 0], line1_real_3d_data[:, 1], line1_real_3d_data[:, 2])

        # line2_sim_3d_data = sim_node.detach().numpy()
        # line2_sim_3d.set_data(line2_sim_3d_data[:, 0], line2_sim_3d_data[:, 1])
        # line2_sim_3d.set_3d_properties(line2_sim_3d_data[:, 2])

        # scatter1_sim_3d._offsets3d = (line2_sim_3d_data[:, 0],
        #                               line2_sim_3d_data[:, 1],
        #                               line2_sim_3d_data[:, 2])

        # line3_loss.set_data(np.arange(len(save_loss_itr)), save_loss_itr)

        # line4_itr_C_dis_max.set_data(np.arange(len(save_C_dis_max_itr)),
        #                              save_C_dis_max_itr)
        # line5_itr_C_bending_max.set_data(
        #     np.arange(len(save_C_bending_max_itr)), save_C_bending_max_itr)
        # line6_itr_C_strain_max.set_data(np.arange(len(save_C_strain_max_itr)),
        #                                 save_C_strain_max_itr)

        plt.show()

    # -------- Save the Optimization Data --------
    def SaveData(self, save_loss_itr, save_curr_itr_time, save_total_itr_time,
                 save_nodes_pos_itr, save_nodes_radius_itr, save_C_dis_itr,
                 save_C_bending_itr, save_C_strain_itr, save_C_radius_itr,
                 save_C_simplevol_itr, save_C_complexvol_itr, node_num,
                 save_dir, num_frame):
        ### ==========================================================================================
        ### SAVE DATA
        save_loss_itr = np.array(save_loss_itr)
        save_curr_itr_time = np.array(save_curr_itr_time)
        save_total_itr_time = np.array(save_total_itr_time)
        save_nodes_pos_itr = np.array(save_nodes_pos_itr)
        save_nodes_radius_itr = np.array(save_nodes_radius_itr)

        min_total_itr_loss_id = np.argmin(
            save_loss_itr,
            axis=0)  ## only first min id is returned by using np.argmin
        #pdb.set_trace()

        # --------- Save the optimation data --------
        save_optimized_data = np.hstack((
            save_nodes_pos_itr[min_total_itr_loss_id],
            save_nodes_radius_itr[min_total_itr_loss_id].reshape(-1, 1),
            np.zeros(
                (node_num, 1)) + save_curr_itr_time[min_total_itr_loss_id],
            np.zeros(
                (node_num, 1)) + save_total_itr_time[min_total_itr_loss_id],
            np.zeros((node_num, 1)) + min_total_itr_loss_id,
            np.zeros((node_num, 1)) + save_loss_itr[min_total_itr_loss_id],
            ## the following 3 are max constraints convergency over each iteration
            np.zeros((node_num, 1)) +
            np.insert(save_C_dis_itr[min_total_itr_loss_id], 29, 0),
            np.zeros((node_num, 1)) +
            np.insert(save_C_bending_itr[min_total_itr_loss_id], 29, [0, 0]),
            np.zeros((node_num, 1)) +
            np.insert(save_C_strain_itr[min_total_itr_loss_id], 29, 0),
            np.zeros((node_num, 1)) +
            save_C_radius_itr[min_total_itr_loss_id].reshape(-1, 1),
            np.insert(save_C_simplevol_itr[min_total_itr_loss_id], 29,
                      0).reshape(-1, 1),
            np.insert(save_C_complexvol_itr[min_total_itr_loss_id], 29,
                      0).reshape(-1, 1)))
        np.save(save_dir + "/frame_{}.npy".format(num_frame),
                save_optimized_data)


## ================================================================================
#   Execute the Backward Process
## ================================================================================


class ExecuteBackward():
    def Execute(self, start_frame, save_dir, LR_RATE, LAYERS, ITERATION,
                Tomas_solve, MPCsteps, traj_type, trajpoint_num, backward,
                compliance):
        ## ================================================================================
        #    Initial the prerequired data
        ## ================================================================================
        SimulationRope = PBDRope(start_frame, save_dir, LR_RATE, LAYERS,
                                 ITERATION, Tomas_solve, MPCsteps, compliance)

        ## ================================================================================
        #    Initial the Traj for the sim node
        ## ================================================================================
        Traj = TrajWaypoints()
        if traj_type == 'heart':
            traj_gt_simnode = Traj.HeartShape(
                trajpoint_num, 28
            )  #The number of desiered point for the actuated point to run
        elif traj_type == 'horse':
            traj_gt_simnode = Traj.HorseShape(
                trajpoint_num, 28
            )  #The number of desiered point for the actuated point to run
        elif traj_type == 'cube':
            traj_gt_simnode = Traj.CubeShape(
                trajpoint_num
            )  #The number of desiered point for the actuated point to run
        elif traj_type == 'Pyramid':
            traj_gt_simnode = Traj.PyramidShape(
                trajpoint_num
            )  #The number of desiered point for the actuated point to run

        # -------- Save traj shape -------
        np.save(save_dir + "/" + traj_type + ".npy", traj_gt_simnode)

        ## ================================================================================
        #    Initial all the constraints
        ## ================================================================================

        # -------- Inital the Distance constraint --------
        Diff_dist = AutoDiffDistance(SimulationRope.goal_dist,
                                     SimulationRope.compliance,
                                     SimulationRope.dt)

        # -------- Inital the stretch constraint --------
        Diff_stretch = AutoDiffStretch(SimulationRope.compliance,
                                       SimulationRope.dt,
                                       SimulationRope.goal_dist)

        #-------- Inital the bending constraint --------
        Diff_bending = AutoDiffBending(SimulationRope.compliance,
                                       SimulationRope.dt,
                                       SimulationRope.DarbouxVector)

        #-------- Inital the vol1 constraint --------
        Diff_vol1 = AutoDiffSimVolume(SimulationRope.compliance,
                                      SimulationRope.dt,
                                      SimulationRope.goal_volume1)

        #-------- Inital the vol2 constraint --------
        Diff_vol2 = AutoDiffComVolume(SimulationRope.compliance,
                                      SimulationRope.dt,
                                      SimulationRope.goal_volume2)
        #pdb.set_trace()
        # #-------- Inital the SimShape constraint --------
        # Diff_shape1 = PBDRope.AutoSimShape(BaxterRope.xp, BaxterRope.ri,
        #                                    BaxterRope.node_num)

        # #-------- Inital the ComShape constraint --------
        # Diff_shape2 = PBDRope.AutoComShape(BaxterRope.xp,
        #                                    BaxterRope.node_radius_goal,
        #                                    BaxterRope.node_quaternion_goal,
        #                                    BaxterRope.node_num - 1)

        #-------- Inital the radius constraint --------
        Diff_radius = AutoDiffRadius(SimulationRope.compliance,
                                     SimulationRope.dt,
                                     SimulationRope.node_radius_goal)

        # #-------- Initial the gravity constraint --------
        # Diff_gravity = PBDRope.AutoDiffGravity()

        # delta_x = []

        for num_frame in range(SimulationRope.start_frame,
                               traj_gt_simnode.shape[0]):

            ## -------------------------------------
            # initialize data for training
            ## -------------------------------------

            if num_frame != SimulationRope.initial_frame:
                SimulationRope.node_pos_curr = torch.as_tensor(
                    np.load(SimulationRope.save_dir +
                            "/frame_{}.npy".format(num_frame -
                                                   1))[:, :3]).float()
                SimulationRope.node_radius_curr = torch.as_tensor(
                    np.load(SimulationRope.save_dir +
                            "/frame_{}.npy".format(num_frame - 1))[:,
                                                                   3]).float()

            ## -------------------------------------
            # initialize data analysis history
            ## -------------------------------------

            save_loss_itr = []
            save_curr_itr_time = []
            save_total_itr_time = []
            save_nodes_pos_itr = []
            save_nodes_radius_itr = []

            ## -------------------------------------
            # Save the max information for each iteration
            ## -------------------------------------
            save_C_dis_max_itr = []
            save_C_bending_max_itr = []
            save_C_strain_max_itr = []
            save_C_simplevol_max_itr = []
            save_C_complexvol_max_itr = []
            save_C_radius_max_itr = []

            ## -------------------------------------
            #Save all the Constraint information for each iteration
            ## -------------------------------------
            save_C_dis_itr = []
            save_C_bending_itr = []
            save_C_strain_itr = []
            save_C_simplevol_itr = []
            save_C_complexvol_itr = []
            save_C_radius_itr = []

            time_begin_frame = time.time()
            SimulationRope.optimizer.zero_grad()

            for itr in range(SimulationRope.ITERATION):
                time_begin_itr = time.time()

                # reset optimizer gradients
                SimulationRope.optimizer.zero_grad()

                ## ================================================================================
                #     Updating dynamics
                ## ================================================================================
                ### --- velocity ----
                # if itr == 0:
                #     dist_velocity[:] = 0
                # else:
                #     dist_velocity = velocity*dt*0.0
                #     # dist_velocity[:, :] += 9.8*(timedp**2)*0.5 * \
                #     #     torch.as_tensor([0.03434638, 0.9867351, 0.15866369])

                #     dist_velocity += damper * velocity / 10 * (dt**2)
                #     dist_velocity[:, :] += 9.8*(dt**2)*0.5 * \
                #         torch.as_tensor([0.0, 1.0, 0.0])
                #     # dist_velocity[:, 1] += 0
                #     dist_velocityITERATION================================================================
                #     Initial quaternion states
                ## ================================================================================
                SimulationRope.node_quaternion_curr = torch.div(
                    SimulationRope.node_quaternion_curr.T,
                    torch.linalg.norm(SimulationRope.node_quaternion_curr,
                                      dim=1)).T
                SimulationRope.node_quaternion_curr = torch.nan_to_num(
                    SimulationRope.node_quaternion_curr, nan=0)

                ## ================================================================================
                #     Initial constraints parameters
                ## ================================================================================
                Lambda_strain = torch.zeros((SimulationRope.node_num - 1, 3))
                Lambda_vol1 = torch.zeros(SimulationRope.node_num)
                Lambda_vol2 = torch.zeros(SimulationRope.node_num - 1)
                Lambda_radius = torch.zeros(SimulationRope.node_num)
                # Lambda_shape1 = 0

                Lambda_dist = torch.zeros((SimulationRope.node_num - 1, 3))
                Lambda_bending = torch.zeros((SimulationRope.node_num - 2, 3))
                # -------- dist_velocity = torch.zeros_like(BaxterRope.node_pos_curr) --------
                #dist_gravity = torch.zeros_like(SimulationRope.node_pos_curr)
                # constraint_q1 = Quaternion([1, 0, 0, 0]).normalised
                # constraint_q2 = Quaternion([1, 0, 0, 0]).normalised

                node_pos_curr = SimulationRope.node_pos_curr.detach()
                node_radius_curr = SimulationRope.node_radius_curr.detach()
                node_quaternion_curr = SimulationRope.node_quaternion_curr.detach(
                )

                ## Set actuation point
                # node_pos_curr[BaxterRope.id_node_actuated, :] = BaxterRope.update_pos_actuated
                # if backward == False:
                #     node_pos_curr[
                #         SimulationRope.
                #         id_node_actuated, :] = SimulationRope.id_node_actuated
                #else:

                #if itr == 2:
                #    pdb.set_trace()
                #-------- MPC Steps data ---------
                temp_mpc_controlpoint = torch.zeros(
                    (SimulationRope.MPCsteps, SimulationRope.node_num,
                     3))  #save the temp mpc data for each step

                loss = 0  # Initialize the loss for MPC
                ## ================================================================================
                #    Main Loops
                ## ================================================================================
                for j in range(SimulationRope.MPCsteps):
                    node_pos_curr[
                        SimulationRope.
                        id_node_actuated, :] = SimulationRope.update_pos_actuated[
                            j]

                    ### ------- Gravity constraints -------
                    # dist_gravity[1:-1, :] = 9.8 * (BaxterRope.dt**
                    #                                2) * 0.5 * torch.as_tensor(
                    #                                    BaxterRope.gravity_dir) * wg

                    # gravity_stiffness, sim_plane_normal_vec_angle = Diff_gravity.ifSimPlaneInGravityDir(
                    #     centerline_3d_gravity, node_pos_curr, BaxterRope.gravity_dir)
                    # dist_gravity[1:-1, :] = 9.8 * (BaxterRope.dt**
                    #                                2) * 0.5 * BaxterRope.gravity_dir * wg * 1 * gravity_stiffness
                    # dist_gravity[1:-1, :] = 9.8 * (
                    #     SimulationRope.dt**2
                    # ) * 0.5 * SimulationRope.gravity_dir * SimulationRope.wg * 1 * 1.0

                    # node_pos_curr = node_pos_curr + (dx_dist * w_dis + dist_gravity * w_gravity) / (w_gravity + w_dis)
                    # node_pos_curr = node_pos_curr + (dist_gravity *
                    #                                  SimulationRope.w_gravity) / 1
                    # node_pos_curr = node_pos_curr + dist_gravity * w_gravity
                    # node_pos_curr = node_pos_curr + dx_dist * w_dis
                    # pdb.set_trace()
                    #print(node_pos_curr[SimulationRope.id_node_actuated, :])

                    for i in range(SimulationRope.LAYERS):
                        #pdb.set_trace()

                        ### ------- TDMAL distance constraints -------
                        if SimulationRope.Tomas_solve == True:
                            dx_dist, Lambda_dist, C_dis = Diff_dist.solveDistConstraintTDMALinearSolver(
                                node_pos_curr, Lambda_dist, SimulationRope.wab,
                                SimulationRope.wab)
                        ### ------- PBD distance constraints -------
                        else:
                            dx_dist, Lambda_dist, C_dis = Diff_dist.solveDistConstraint(
                                node_pos_curr, Lambda_dist, SimulationRope.wab,
                                SimulationRope.wab)

                        # node_pos_curr = node_pos_curr + dx_dist * w_dis

                        ### ------- Stretch constraint -------
                        dx_strain, Lambda_strain, dq_strain, C_strain = Diff_stretch.C_strain_solve(
                            node_pos_curr, Lambda_strain, node_quaternion_curr,
                            SimulationRope.ws, SimulationRope.ws,
                            SimulationRope.wq, i)

                        # -------- The point in the two sides is fixed and the control point fixed --------
                        dx_strain[0] = dx_strain[0] * 0
                        dx_strain[SimulationRope.id_node_actuated] = dx_strain[
                            SimulationRope.id_node_actuated] * 0
                        dx_strain[-1] = dx_strain[-1] * 0
                        dx_dist[0] = 0 * dx_dist[0]
                        dx_dist[SimulationRope.id_node_actuated] = dx_dist[
                            SimulationRope.id_node_actuated] * 0
                        dx_dist[-1] = 0 * dx_dist[-1]

                        # node_pos_curr = node_pos_curr + dx_strain * w_strain
                        # node_pos_curr = node_pos_curr + (dx_strain * w_strain + dx_dist * w_dis) / 2

                        ### ------- Bending constraint -------
                        dq_bending, Lambda_bending, C_bending = Diff_bending.solve_BendTwistConstraint(
                            node_pos_curr, node_quaternion_curr,
                            Lambda_bending, SimulationRope.wq,
                            SimulationRope.wq)

                        ## ------- Radius constraint -------
                        dr_radius, Lambda_radius, C_radius = Diff_radius.C_radius_solve(
                            Lambda_radius, node_radius_curr)

                        ### ------- Distance constraint -------
                        # dx_dist, Lambda_dist = Diff_dist.solveDistConstraint(node_pos_curr, Lambda_dist, wa * scale_wab,
                        #                                                      wb * scale_wab)
                        # dx_dist, Lambda_dist = Diff_dist.solveDistConstraintTDMALinearSolver(
                        #     node_pos_curr, Lambda_dist, wa * scale_wab, wb * scale_wab)

                        ## -------- Volume constraint --------
                        _, dr_vol1, Lambda_vol1, C_simplevol = Diff_vol1.solveVolumeConstraint(
                            node_pos_curr, node_radius_curr, Lambda_vol1)

                        ## -------- Volume2 constraint --------
                        _, dr_vol2, Lambda_vol2, C_complexvol = Diff_vol2.solveVolumeConstraint(
                            node_pos_curr, node_radius_curr, Lambda_vol2)
                        #pdb.set_trace()

                        ### ------- Shape matching constraint -------
                        # dx_shapesim, constraint_q1, L = Diff_shape1.solveShapeMatchingConstraint(
                        #     node_pos_curr, node_radius_curr, constraint_q1, L)
                        # dx_com, dr_com, dq_com, constraint_q2 = Diff_shape2.solveShapeMatchingConstraint(
                        #     node_pos_curr, node_quaternion_curr, node_radius_curr, constraint_q2)

                        ### ==========================================================================================
                        ###       Updating states
                        ### ==========================================================================================
                        ### position
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_dist + dx_strain * 6.0 + dist_gravity) / 8.0 * 1
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_strain * 6.0 + dist_gravity) / 7.0 * 1
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_strain * w_strain) / 1.0
                        # node_pos_curr = node_pos_curr + dx_strain * w_strain

                        #-------- Update the position of the simulation node --------
                        node_pos_curr = node_pos_curr + (
                            dx_strain * SimulationRope.w_strain + dx_dist *
                            SimulationRope.w_dis) / 2 * SimulationRope.w_SOR
                        # node_pos_curr = node_pos_curr + (dx_strain * w_strain + dx_dist * w_dis) / (w_strain + w_dis)

                        ## -------- quaternion --------
                        node_quaternion_curr = node_quaternion_curr + (
                            dq_strain * SimulationRope.wq_strain +
                            dq_bending * SimulationRope.wq_bending
                        ) / 2 * SimulationRope.w_SOR
                        # node_quaternion_curr = node_quaternion_curr + (dq_strain * wq_strain +
                        #                                                dq_bending * wq_bending) / (wq_strain + wq_bending)

                        node_quaternion_curr = torch.div(
                            node_quaternion_curr.T,
                            torch.linalg.norm(node_quaternion_curr, dim=1)).T
                        node_quaternion_curr = torch.nan_to_num(
                            node_quaternion_curr, nan=0)

                        #-------- Update the radius --------
                        node_radius_curr = node_radius_curr + (
                            dr_vol1 + dr_vol2 + dr_radius) / 3
                        #pdb.set_trace()
                    ### ==========================================================================================
                    ###       To judge the dimension of the traj
                    ### ==========================================================================================
                    if traj_type == 'heart':  # Heart is two dimension

                        node_pos_curr[:, 2] = 0
                        #pdb.set_trace()

                    ### ==========================================================================================
                    ###       Updating BaxterRope related information
                    ### ==========================================================================================
                    SimulationRope.node_pos_curr = node_pos_curr
                    SimulationRope.node_quaternion_curr = node_quaternion_curr
                    SimulationRope.node_radius_curr = node_radius_curr
                    temp_mpc_controlpoint[j] = node_pos_curr

                    ### ==========================================================================================
                    ##         Get different loss
                    ### ==========================================================================================
                    # loss_projection_lineseg_3D = BaxterRope.getProjectionLineSegmentsLoss(
                    #     node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                    #     node_pos_curr, centerline_3d_gravity)
                    # --------- Calculate the Loss --------
                    loss = loss + torch.linalg.norm(
                        node_pos_curr[SimulationRope.id_node_target] -
                        traj_gt_simnode[num_frame])

                ### ==========================================================================================
                ##         print related backward result
                ### ==========================================================================================
                #pdb.set_trace()
                time_finish_frame = time.time()
                curr_itr_time = time_finish_frame - time_begin_itr
                total_itr_time = time_finish_frame - time_begin_frame
                print("\n")
                print("itr", itr)
                print("Curr  itr time :", curr_itr_time)
                print("Total itr time :", total_itr_time)
                print("The loss is    :", loss)
                print(
                    "---------------------------------------------------------------------------------------------"
                )

                print(
                    "---------------------------------------------------------------------------------------------"
                )
                print("\n")

                ### ==========================================================================================
                ##         Constraint Losses
                ### ==========================================================================================

                C_dis_max = torch.max(torch.abs(C_dis))
                C_bending_max = torch.max(torch.linalg.norm(C_bending, dim=1))
                C_strain_max = torch.max(torch.linalg.norm(C_strain, dim=1))
                C_radius_max = torch.max(torch.linalg.norm(C_radius, dim=0))
                C_simplevol_max = torch.max(C_simplevol, dim=0)
                C_complexvol_max = torch.max(C_complexvol, dim=0)

                ### ==========================================================================================
                ### Do loss backpropagation
                ### ==========================================================================================
                # loss.backward(retain_graph=True)
                if itr == 200:
                    print(node_pos_curr[SimulationRope.id_node_actuated])
                if backward == True:
                    loss.backward()
                    SimulationRope.optimizer.step()
                    if itr == 200:
                        pdb.set_trace()

                print("The gradient of control point:",
                      SimulationRope.update_pos_actuated.grad)
                print(
                    "The position of target point:", node_pos_curr[
                        SimulationRope.id_node_target].detach().numpy())
                print("The ground true target point:",
                      traj_gt_simnode[num_frame].detach().numpy())
                #pdb.set_trace()

                ### ==========================================================================================
                ### Recording
                #pdb.set_trace()

                # --------- Save the information for loss adn time and simulation node -------
                save_loss_itr.append(loss.detach().numpy())
                save_curr_itr_time.append(curr_itr_time)
                save_total_itr_time.append(total_itr_time)
                save_nodes_pos_itr.append(
                    temp_mpc_controlpoint[0].detach().numpy())
                save_nodes_radius_itr.append(
                    SimulationRope.node_radius_curr.detach().numpy())

                # --------- Save the information for constraints with max offset-------
                #pdb.set_trace()
                save_C_dis_max_itr.append(C_dis_max.detach().numpy())
                save_C_bending_max_itr.append(C_bending_max.detach().numpy())
                save_C_strain_max_itr.append(C_strain_max.detach().numpy())
                save_C_radius_max_itr.append(C_radius_max.detach().numpy())
                #pdb.set_trace()
                save_C_simplevol_max_itr.append(
                    C_simplevol_max.values.detach().numpy())
                #pdb.set_trace()
                save_C_complexvol_max_itr.append(
                    C_complexvol_max.values.detach().numpy())

                # --------- Save the information for constraints for each iteration-------
                save_C_dis_itr.append(C_dis.detach().numpy())
                save_C_bending_itr.append(C_bending.detach().numpy())
                save_C_strain_itr.append(C_strain.detach().numpy())
                save_C_radius_itr.append(C_radius.detach().numpy())
                save_C_simplevol_itr.append(C_simplevol.detach().numpy())
                save_C_complexvol_itr.append(C_complexvol.detach().numpy())
                #pdb.set_trace()

            ### ==========================================================================================
            ### Processingv optimization Data
            OptimizedData = ProcessOptData()

            ### ==========================================================================================
            # ## Plotting
            # OptimizedData.ShowImage(SimulationRope.ITERATION, Traj_sim_node,
            #                         node_pos_curr, save_loss_itr,
            #                         save_C_dis_max_itr, save_C_bending_max_itr,
            #                         save_C_strain_max_itr)
            #plt.pause(1.0)
            #pdb.set_trace()
            ## ==========================================================================================
            ### Save Data
            OptimizedData.SaveData(
                save_loss_itr, save_curr_itr_time, save_total_itr_time,
                save_nodes_pos_itr, save_nodes_radius_itr, save_C_dis_itr,
                save_C_bending_itr, save_C_strain_itr, save_C_radius_itr,
                save_C_simplevol_itr, save_C_complexvol_itr,
                SimulationRope.node_num, save_dir, num_frame)
