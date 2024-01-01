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
    def __init__(self, start_frame, end_frame, save_dir, LR_RATE, LAYERS,
                 STEPS, ITERATION):

        # --------- define number of particles ---------
        self.node_num = 20
        self.compliance = 2e-6
        self.dt = 0.1
        self.damper = 0.0

        # --------- Initialize Frame infomation ---------
        self.initial_frame = 14
        self.start_frame = start_frame  #54
        self.end_frame = end_frame

        # --------- Initialize carmera, gravity parameter ---------
        self.cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])
        gravity_dir = torch.as_tensor([0.10381715, 0.9815079, -0.16082364])
        self.gravity_dir = gravity_dir / torch.linalg.norm(gravity_dir, dim=0)

        # --------- Judge save dir exits or not ---------
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # --------- Initialize Backward Information ---------
        self.LR_RATE = LR_RATE
        # self.LAYERS = 10
        # self.STEPS = 50
        self.LAYERS = LAYERS
        self.STEPS = STEPS
        self.ITERATION = ITERATION

        # --------- Initialize Goal State ---------
        self.setGoalState()

        # --------- Initialize  ---------
        self.setInitialization()

        # --------- Actuated/Target ID ---------
        id_node_target = []
        id_node_actuated = []
        id_node_fixed = []

        for i in range(self.node_num):
            if i != 0:
                id_node_target.append(i)
            elif i == 0:
                id_node_actuated.append(i)

        # --------- Initialize ID ---------
        self.id_node_target = torch.as_tensor(id_node_target)
        self.id_node_actuated = torch.as_tensor(id_node_actuated)
        self.id_node_fixed = torch.as_tensor(id_node_fixed)

        # --------- Weight for the Constraint ---------
        self.wab = 1.0
        self.wg = 1.0
        self.ws = 1.0
        self.wq = 1.0

        # ### ------- Constraints stiffness -------
        # self.w_gravity = 0.20070688

        # self.w_dis = 0.87617564

        # self.w_strain = 1.1042638  ## effect : if on the same plane

        # self.wq_strain = 0.8733262

        # self.wq_bending = 1.0066655  ## effect : curvature

        # self.w_SOR = 1.3067461
        ## ------- Constraints stiffness -------
        self.w_gravity = 1.2101464 / self.STEPS

        self.w_dis = 0.48465276

        self.w_strain = 1.1617689  # effect : if on the same plane

        self.wq_strain = 0.5212269

        self.wq_bending = 1.4967155

        # effect : curvature

        self.w_SOR = 0.6120443

        # ------- Variable and Optimizer -------
        self.update_pos_actuated = Variable(self.node_pos_curr[0].clone(),
                                            requires_grad=True)
        self.optimizer = optim.Adam([self.update_pos_actuated],
                                    lr=self.LR_RATE,
                                    weight_decay=0.000)
        torch.autograd.set_detect_anomaly(True)

    def setGoalState(self):

        # define goal position of particles
        # goal_pos = torch.stack(
        #    [torch.arange(self.node_num), 2 * torch.arange(self.node_num), 3 * torch.arange(self.node_num)])

        # self.node_pos_goal = torch.transpose(goal_pos, 0, 1)

        # ------- Initialize Goal Position for the node -------
        self.node_pos_goal = torch.zeros((self.node_num, 3))
        centerline_3d_gravity = np.load(
            "../DATA/DATA_BaxterRope/downsampled_pcl/initialization.npy") * 1.0
        min = centerline_3d_gravity[0]
        max = centerline_3d_gravity[-1]

        distance = (max - min) / (self.node_num - 1)
        for i in range(self.node_num):
            self.node_pos_goal[i] = torch.as_tensor(min + i * distance) * 1.0

        # ------- Initialize Goal Radius for the node -------
        self.node_radius_goal = torch.ones(self.node_num)
        for i in range((self.node_num)):
            self.node_radius_goal[i] = 0.01

        # ------- Initialize Rest Lenght for the node -------
        self.goal_dist = self.RestLength(self.node_pos_goal)
        # self.goal_dist[:] = 0.0571
        # self.goal_dist[:] = 0.0601

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
        initial_states = torch.as_tensor(
            np.load(self.save_dir + "/frame_{}_init.npy".format(
                self.initial_frame))).float() * 1
        self.node_pos_curr = initial_states[:, 0:3]
        #pdb.set_trace()

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
        dis_pos = torch.linalg.norm(diff_pos, ord=None, dim=1)
        diff_radius = torch.diff(radius, dim=0)
        e = -diff_radius / dis_pos

        L = dis_pos + diff_radius * e

        vol = math.pi / 3.0 * (
            torch.diff(radius**3, dim=0) * (e**3 - 3.0 * e) + L *
            (1.0 - e**2) * (radius_a**2 + radius_a * radius_b + radius_b**2))
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
#   Stretch Constraint
## ================================================================================


class AutoDiffStretch():
    def __init__(self, compliance, dt, rest_length):
        self.compliance = compliance
        self.dt = dt
        self.rest_length = rest_length

    def C_strain_solve(self, pos, Lamda, quat, wa, wb, wq):

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
        Costraint = diff_pos_norm - self.rest_length[:]
        # print(c)
        # pdb.set_trace()

        dL = torch.sub(-alpha * Lamda.T, Costraint)
        dL = dL.T / (wa + wb + alpha)
        delta_x = torch.mul(dL, diff_pos_normalized)
        Lamda += dL
        dx[:-1] += delta_x * wa
        dx[1:] -= delta_x * wb
        # pdb.set_trace()
        return dx, Lamda

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

        # pdb.set_trace()

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

        # pdb.set_trace()

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
        X = torch.mul(r3ab.T, s * s).T - self.Vi
        W = X
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
        return dx, dr, lamda


## ================================================================================
#   Complex Volume Constraint
## ================================================================================


class AutoDiffComVolume():
    def __init__(self, compliance, dt, volume):
        self.compliance = compliance
        self.dt = dt
        self.volume = volume

    def solveVolumeConstraint(self, pos, radius):
        dx = torch.zeros((pos.shape[0], 3))
        dr = torch.zeros((pos.shape[0]))
        Lambda = torch.zeros(pos.shape[0] - 1)
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

        c = torch.div(V, self.volume) - 1.0

        alpha = self.compliance / (self.dt * self.dt)
        dL = torch.sub(-alpha * Lambda, c)

        dL = dL / (wSum + alpha)

        dpa = torch.mul(ban.T, (gpa_n * dL)).T
        # print(dpa)
        dra = gra * dL
        drb = grb * dL

        dx[:-1] += dpa
        dx[1:] -= dpa

        dr[:-1] += dra
        dr[1:] += drb

        return dx, dr, Lambda


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
        c = torch.div(r_cur, self.r) - 1.0
        dL = (-c - alpha * L) / (1 / (self.r * self.r) + alpha)
        drr = 1 / self.r * dL
        dr = drr
        L += dL

        return dr, L


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
#  Gravity Constraint
## ================================================================================


class AutoDiffGravity():
    def ProjectionPointAlongGravity(self, centerline_3d_gravity,
                                    node_pos_curr):

        extend_gravity_scale = 1
        gravity_dir = torch.as_tensor([0.10381715, 0.9815079, -0.16082364])
        gravity = torch.vstack(
            (centerline_3d_gravity[0] - gravity_dir * extend_gravity_scale,
             centerline_3d_gravity[0] + gravity_dir * extend_gravity_scale))
        gravity_vector = gravity[1] - gravity[0]
        gravity_length_square = torch.linalg.norm(gravity_vector)**2

        sim_proj_gravity = torch.sum(
            (node_pos_curr - gravity[0]) * gravity_vector, dim=1)
        sim_proj_gravity = sim_proj_gravity / gravity_length_square
        sim_lowest_gravity_id = torch.max(sim_proj_gravity, dim=0)
        sim_lowest_gravity = node_pos_curr[sim_lowest_gravity_id.indices]

        sim_highest_gravity_id = torch.min(sim_proj_gravity, dim=0)
        sim_highest_gravity = node_pos_curr[sim_highest_gravity_id.indices]

        return sim_lowest_gravity, sim_highest_gravity

    def NormalVectorForSimPlane(self, centerline_3d_gravity, node_pos_curr,
                                itr):
        sim_lowest_gravity, sim_highest_gravity = self.ProjectionPointAlongGravity(
            centerline_3d_gravity, node_pos_curr)

        #The normal vector combination for sim_node
        normalvector1 = sim_lowest_gravity - node_pos_curr[3]
        normalvector2 = sim_lowest_gravity - node_pos_curr[-2]
        normalvector1 = normalvector1 / torch.linalg.norm(normalvector1)
        normalvector2 = normalvector2 / torch.linalg.norm(normalvector2)
        NormalVector = torch.cross(normalvector1, normalvector2)
        NormalVector = NormalVector / torch.linalg.norm(NormalVector)

        dot_product = np.dot(NormalVector, gravity_dir)
        noraml_venctor_angle = np.arccos(dot_product)

        gravity_stiffness = abs(
            torch.cos(torch.as_tensor(noraml_venctor_angle))) * 1

        return gravity_stiffness, noraml_venctor_angle


## ================================================================================
#   Process the optimization Data
## ================================================================================


class ProcessOptData():
    # -------- Show the Image --------
    def ShowImage(self, ITERATION, centerline_3d_gravity, sim_node,
                  save_loss_itr, save_C_dis_max_itr, save_C_bending_max_itr,
                  save_C_strain_max_itr):
        print("==========================================================")

        print("\n")
        fig = plt.figure(figsize=(15, 8))
        # ax = axes.ravel()
        ax_3d_curve = fig.add_subplot(2, 3, 1, projection='3d')
        # init_plot_data = np.array([0, 0, 0])
        line1_real_3d, = ax_3d_curve.plot([], [],
                                          color='#8bc24c',
                                          linestyle='-',
                                          linewidth=2)
        line2_sim_3d, = ax_3d_curve.plot([], [],
                                         color='#de4307',
                                         linestyle='--',
                                         linewidth=1)
        scatter1_sim_3d = ax_3d_curve.scatter([], [], [],
                                              s=16,
                                              c='#8134af',
                                              alpha=0.8)
        # line3, = ax.plot(0, 0, color='#1F6ED4', marker='o', markersize=6)
        ax_3d_curve.set_xlim3d([-0.4, 0.5])
        ax_3d_curve.set_xlabel('X')
        ax_3d_curve.set_ylim3d([0, 0.5])
        ax_3d_curve.set_ylabel('Y')
        ax_3d_curve.set_zlim3d([0.6, 1.0])
        ax_3d_curve.set_zlabel('Z')

        ax_loss = fig.add_subplot(2, 3, 2)
        line3_loss, = ax_loss.plot([], [],
                                   color='#1F6ED4',
                                   marker='o',
                                   markersize=3,
                                   linestyle='-',
                                   linewidth=1)
        ax_loss.set_xlim(0, ITERATION)
        ax_loss.set_ylim(0, 100)
        ax_loss.set_xlabel('iter (steps)')  # Add an x-label to the axes.
        ax_loss.set_ylabel('loss')
        ax_loss.set_title('loss over frames')

        ax_itr_C_dis_max = fig.add_subplot(2, 3, 4)
        line4_itr_C_dis_max, = ax_itr_C_dis_max.plot([], [],
                                                     color='#1F6ED4',
                                                     marker='o',
                                                     markersize=3,
                                                     linestyle='-',
                                                     linewidth=1)
        ax_itr_C_dis_max.set_xlim(0, ITERATION)
        ax_itr_C_dis_max.set_ylim(0, 0.1)
        ax_itr_C_dis_max.set_xlabel(
            'iter (steps)')  # Add an x-label to the axes.
        ax_itr_C_dis_max.set_ylabel('max C_dis')
        ax_itr_C_dis_max.set_title('Max C_dis constraint convergency')

        ax_itr_C_bending_max = fig.add_subplot(2, 3, 5)
        line5_itr_C_bending_max, = ax_itr_C_bending_max.plot([], [],
                                                             color='#1F6ED4',
                                                             marker='o',
                                                             markersize=3,
                                                             linestyle='-',
                                                             linewidth=1)
        ax_itr_C_bending_max.set_xlim(0, ITERATION)
        ax_itr_C_bending_max.set_ylim(0, 0.1)
        ax_itr_C_bending_max.set_xlabel(
            'iter (steps)')  # Add an x-label to the axes.
        ax_itr_C_bending_max.set_ylabel('max C_bending')
        ax_itr_C_bending_max.set_title('Max C_bending constraint convergency')

        ax_itr_C_strain_max = fig.add_subplot(2, 3, 6)
        line6_itr_C_strain_max, = ax_itr_C_strain_max.plot([], [],
                                                           color='#1F6ED4',
                                                           marker='o',
                                                           markersize=3,
                                                           linestyle='-',
                                                           linewidth=1)
        ax_itr_C_strain_max.set_xlim(0, ITERATION)
        ax_itr_C_strain_max.set_ylim(0, 2)
        ax_itr_C_strain_max.set_xlabel(
            'iter (steps)')  # Add an x-label to the axes.
        ax_itr_C_strain_max.set_ylabel('max C_strain')
        ax_itr_C_strain_max.set_title('Max C_strain constraint convergency')

        #def ShowImage(self, centerline_3d_gravity, sim_node):
        line1_real_3d_data = centerline_3d_gravity.detach().numpy()
        line1_real_3d.set_data(line1_real_3d_data[:, 0], line1_real_3d_data[:,
                                                                            1])
        line1_real_3d.set_3d_properties(line1_real_3d_data[:, 2])

        # scatter1_sim_3d._offsets3d = (line1_real_3d_data[:, 0], line1_real_3d_data[:, 1], line1_real_3d_data[:, 2])

        line2_sim_3d_data = sim_node.detach().numpy()
        line2_sim_3d.set_data(line2_sim_3d_data[:, 0], line2_sim_3d_data[:, 1])
        line2_sim_3d.set_3d_properties(line2_sim_3d_data[:, 2])

        scatter1_sim_3d._offsets3d = (line2_sim_3d_data[:, 0],
                                      line2_sim_3d_data[:, 1],
                                      line2_sim_3d_data[:, 2])

        line3_loss.set_data(np.arange(len(save_loss_itr)), save_loss_itr)

        line4_itr_C_dis_max.set_data(np.arange(len(save_C_dis_max_itr)),
                                     save_C_dis_max_itr)
        line5_itr_C_bending_max.set_data(
            np.arange(len(save_C_bending_max_itr)), save_C_bending_max_itr)
        line6_itr_C_strain_max.set_data(np.arange(len(save_C_strain_max_itr)),
                                        save_C_strain_max_itr)

        plt.show()

    # -------- Save the Optimization Data --------
    def SaveData(self, save_loss_itr, save_curr_itr_time, save_total_itr_time,
                 save_nodes_pos_itr, save_nodes_radius_itr,
                 save_nodes_quaternion_itr, C_dis_itrs, C_bending_itrs,
                 C_strain_itrs, node_num, save_dir, num_frame):
        ### ==========================================================================================
        ### SAVE DATA
        save_loss_itr = np.array(save_loss_itr)
        save_curr_itr_time = np.array(save_curr_itr_time)
        save_total_itr_time = np.array(save_total_itr_time)
        save_nodes_pos_itr = np.array(save_nodes_pos_itr)
        save_nodes_radius_itr = np.array(save_nodes_radius_itr)
        save_nodes_quaternion_itr = np.array(save_nodes_quaternion_itr)
        C_dis_itrs = np.array(C_dis_itrs)
        C_bending_itrs = np.array(C_bending_itrs)
        C_strain_itrs = np.array(C_strain_itrs)

        min_total_itr_loss_id = np.argmin(
            save_loss_itr,
            axis=0)  ## only first min id is returned by using np.argmin

        final_quaternion = np.zeros((node_num, 4))
        final_quaternion[:node_num - 1, :] = save_nodes_quaternion_itr[
            min_total_itr_loss_id]
        save_optimized_data = np.hstack((
            # -------- Sim Node information -------
            save_nodes_pos_itr[min_total_itr_loss_id],
            save_nodes_radius_itr[min_total_itr_loss_id].reshape(-1, 1),
            final_quaternion,

            # -------- Constraint information --------
            np.insert(C_dis_itrs[min_total_itr_loss_id], 19, 0).reshape(-1, 1),
            np.insert(C_bending_itrs[min_total_itr_loss_id], 18,
                      [0, 0]).reshape(-1, 1),
            np.insert(C_strain_itrs[min_total_itr_loss_id], 19,
                      0).reshape(-1, 1),

            # -------- loss and time information -------
            np.zeros((node_num, 1)) + min_total_itr_loss_id,
            np.zeros((node_num, 1)) + save_loss_itr[min_total_itr_loss_id],
            np.zeros(
                (node_num, 1)) + save_curr_itr_time[min_total_itr_loss_id],
            np.zeros(
                (node_num, 1)) + save_total_itr_time[min_total_itr_loss_id],
        ))
        #pdb.set_trace()
        np.save(save_dir + "/frame_{}.npy".format(num_frame),
                save_optimized_data)

    # -------- Save temp data --------
    def SaveTempData(self, loss_temp, C_dis, C_strain, C_bending, itr, j, i,
                     cal_loss_method, cal_loss_target):

        Save_Temp_Data = np.hstack(
            (np.insert(C_dis, 19, loss_temp),
             np.insert(np.linalg.norm(C_strain, axis=1), 19, 0),
             np.insert(np.linalg.norm(C_bending, axis=1), 18, [0, 0])))
        #pdb.set_trace()

        np.save(
            "../DATA/CompareOverXPBDAndTomas/Tomas_30/" + cal_loss_method +
            "_" + cal_loss_target + "_" + cal_loss_target + str(itr) + "_" +
            str(j) + "_" + str(i) + ".npy", Save_Temp_Data)


## ================================================================================
#   The function combines the function need to calculate the final loss
## ================================================================================


class LossList():
    def __init__(self, cam_intrinsic_mat):
        self.cam_intrinsic_mat = cam_intrinsic_mat

    ## ================================================================================
    #   For all the point cloud, we need to find the closet distance to all the 19 segment lines
    ## ================================================================================
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
    #   Get the corresponding point to point loss
    ## ================================================================================
    def getP2PCorrespondLoss(self,
                             gt_centerline,
                             node_pos_curr,
                             two_dimension=True):

        # Note: this function is applicable to 2d and 3d centerline projection

        if two_dimension:
            pt_3d_project = torch.matmul(self.cam_intrinsic_mat,
                                         node_pos_curr.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]
        else:
            sim_pbd = node_pos_curr

        real_gt = torch.as_tensor(gt_centerline)
        real_gt_diff = torch.diff(real_gt, dim=0)
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

        real_gt_correpond = real_gt[real2sim_id.indices]
        loss_correpond = torch.sum(
            torch.linalg.norm(sim_pbd - real_gt_correpond, dim=1))

        return loss_correpond, real2sim_id.indices

    ## ================================================================================
    #   Get LowestP Point Along y axis
    ## ================================================================================
    def getLowestPointAlongYLoss(self, centerline_3d_gravity, node_pos_curr):

        ## find the lowested along Y : sim PBD node
        sim_y_lowest_id = torch.max(node_pos_curr[:, 1], dim=0)
        sim_y_lowest = node_pos_curr[sim_y_lowest_id.indices]

        ## find the lowested along Y : centerline_3d_gravity
        real_y_lowest_id = torch.max(centerline_3d_gravity[:, 1], dim=0)
        real_y_lowest = centerline_3d_gravity[real_y_lowest_id.indices]

        loss_lowest_point_gravity = torch.linalg.norm(real_y_lowest -
                                                      sim_y_lowest)

        return loss_lowest_point_gravity

    ## ================================================================================
    #   Get LowestP Point Along gravity direction
    ## ================================================================================
    def getLowestPointAlongGravityLoss(self, centerline_3d_gravity,
                                       node_pos_curr):

        extend_gravity_scale = 1
        gravity_dir = torch.as_tensor([0.10381715, 0.9815079, -0.16082364])
        gravity = torch.vstack(
            (centerline_3d_gravity[0] - gravity_dir * extend_gravity_scale,
             centerline_3d_gravity[0] + gravity_dir * extend_gravity_scale))
        gravity_vector = gravity[1] - gravity[0]
        gravity_length_square = torch.linalg.norm(gravity_vector)**2

        # The real lowest point along the gravity
        real_proj_gravity = torch.sum(
            (centerline_3d_gravity - gravity[0]) * gravity_vector, dim=1)
        real_proj_gravity = real_proj_gravity / gravity_length_square
        real_lowest_gravity_id = torch.max(real_proj_gravity, dim=0)
        real_lowest_gravity = centerline_3d_gravity[
            real_lowest_gravity_id.indices]

        # The simulation node's lowest point along the gravity
        sim_proj_gravity = torch.sum(
            (node_pos_curr - gravity[0]) * gravity_vector, dim=1)
        sim_proj_gravity = sim_proj_gravity / gravity_length_square
        sim_lowest_gravity_id = torch.max(sim_proj_gravity, dim=0)
        sim_lowest_gravity = node_pos_curr[sim_lowest_gravity_id.indices]

        # # The simulation node's highest point along the gravity
        # sim_highest_gravity_id = torch.min(sim_proj_gravity, dim=0)
        # sim_highest_gravity = node_pos_curr[sim_highest_gravity_id.indices]

        loss_lowest_point_gravity = torch.linalg.norm(real_lowest_gravity -
                                                      sim_lowest_gravity)

        # pdb.set_trace()

        return loss_lowest_point_gravity, real_lowest_gravity, sim_lowest_gravity

    ## ================================================================================
    #   Get the 2D projection point from 3D simulation point
    ## ================================================================================
    def get2DProjectionPointfrom3DSimPoints(self, sim_nodes):
        projectpoint_2d_simnode = torch.matmul(self.cam_intrinsic_mat,
                                               sim_nodes.float().T).T
        projectpoint_2d_simnode_normalized = torch.div(
            projectpoint_2d_simnode,
            (projectpoint_2d_simnode[:, 2]).reshape(-1, 1))
        return projectpoint_2d_simnode_normalized[:, :2]

    ## ================================================================================
    #   Get the point to point loss
    ## ================================================================================
    def getP2PLoss(self, gt_centerline, node_pos_curr, index_real2sim__SimNode,
                   two_dimension):
        _, real2sim_IdIndices_GtCenterline = self.getP2PCorrespondLoss(
            gt_centerline, node_pos_curr, two_dimension)
        real2sim_GtCenterline_index = real2sim_IdIndices_GtCenterline[
            index_real2sim__SimNode]
        real2sim_GtCenterline_ptp = gt_centerline[real2sim_GtCenterline_index]
        real2sim_SimNode_ptp = node_pos_curr[index_real2sim__SimNode]
        P2PLoss = torch.sum(
            torch.linalg.norm(real2sim_GtCenterline_ptp - real2sim_SimNode_ptp,
                              dim=1))
        return P2PLoss

    ## ================================================================================
    #   Get the loss for the difference of normal vector between sim node and real ndoe on centerline
    ## ================================================================================
    def getNormalVectorLossforSimAndReal(self, centerline_3d_gravity,
                                         node_pos_curr):
        _, real_lowest_gravity, sim_lowest_gravity = self.getLowestPointAlongGravityLoss(
            centerline_3d_gravity, node_pos_curr)

        #Get the Normal Vector for SimNode
        curr_NormalVector_1 = node_pos_curr[-2] - sim_lowest_gravity
        curr_NormalVector_2 = node_pos_curr[3] - sim_lowest_gravity
        curr_normal = torch.cross(curr_NormalVector_1, curr_NormalVector_2)
        curr_normal = curr_normal.detach().numpy()
        curr_normal_normalized = curr_normal / np.linalg.norm(curr_normal)

        #Get the Normal Vector for CenterNode
        center_NormalVector_1 = centerline_3d_gravity[0] - real_lowest_gravity
        center_NormalVector_2 = centerline_3d_gravity[-1] - real_lowest_gravity
        center_normal = torch.cross(center_NormalVector_1,
                                    center_NormalVector_2)
        center_normal = center_normal.detach().numpy()
        center_normal_normalized = center_normal / np.linalg.norm(
            center_normal)

        #Get the angle between the two vector
        curr_center_dot_product = np.dot(curr_normal_normalized[0],
                                         center_normal_normalized[0])
        normalvector_angle = np.arccos(curr_center_dot_product)
        normalVector_loss = torch.as_tensor(normalvector_angle / math.pi)
        return normalVector_loss

    ## ================================================================================
    #   Get the loss for the difference of lenght between sim node and real ndoe on centerline
    ## ================================================================================
    def getLossForLengthBetweenSimAndReal(self, centerline_3d_gravity,
                                          node_pos_curr):
        centerline_3d_gravity_diff = torch.diff(centerline_3d_gravity, dim=0)
        length_centerline_3d_gravity = torch.sum(
            torch.linalg.norm(centerline_3d_gravity_diff, dim=1))

        node_pos_curr_diff = torch.diff(node_pos_curr, dim=0)
        node_pos_curr_length = torch.sum(
            torch.linalg.norm(node_pos_curr_diff, dim=1))

        loss_3d_lenght = abs(node_pos_curr_length -
                             length_centerline_3d_gravity)
        #pdb.set_trace()

        return loss_3d_lenght

    ## ================================================================================
    #  Get corresponding point to point loss
    ## ================================================================================
    def getP2PCorrespondLoss(self,
                             gt_centerline,
                             node_pos_curr,
                             two_dimension=True):

        if two_dimension:
            pt_3d_project = torch.matmul(self.cam_intrinsic_mat,
                                         node_pos_curr.float().T).T
            sim_pt_2d = torch.div(pt_3d_project,
                                  (pt_3d_project[:, 2]).reshape(-1, 1))
            sim_pbd = sim_pt_2d[:, :2]
        else:
            sim_pbd = node_pos_curr

        real_gt = torch.as_tensor(gt_centerline)
        real_gt_diff = torch.diff(real_gt, dim=0)
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
#   The function to calcualte the loss
## ================================================================================
class GetLoss():
    def CalculateLoss(self, cal_loss_method, cal_loss_target, node_pos_curr,
                      num_frame, cam_intrinsic_mat):

        # Read Before run
        # Notice:  cal_loss_method : projection,correspondance,segments_projection,chamfer distance
        # cal_loss_target : centerline_2d,pt_lwst_3Dcenterline,centerline_3d_gravity,3D_3pt,3D_20pt,//
        # centerline_2d_plus_lwst,centerline_2d_lwst_normal_vector,centerline_3d_gravity_plus_length//
        #centerline_3d_gravity_plus_normalvector

        if cal_loss_method == "projection":
            loss = self.getProjectionLoss(cal_loss_target, node_pos_curr,
                                          num_frame, cam_intrinsic_mat)

        if cal_loss_method == "correspondance":
            loss = self.getCorrespondanceLoss(cal_loss_target, node_pos_curr,
                                              num_frame, cam_intrinsic_mat)

        if cal_loss_method == "seg":
            loss = self.getSegmentsProjectionLoss(cal_loss_target,
                                                  node_pos_curr, num_frame,
                                                  cam_intrinsic_mat)

        return loss

    ## ================================================================================
    #   The function to calcualte the loss for projection
    ## ================================================================================
    def getProjectionLoss(self, cal_loss_target, node_pos_curr, num_frame,
                          cam_intrinsic_mat):

        #--------- load all loss type ---------
        LossTypeList = LossList(cam_intrinsic_mat)

        #---------  Load the centerline --------
        centerline_3d_gravity = torch.as_tensor(
            np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)).float()
        centerline_2d_image = torch.as_tensor(
            np.load("../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy" %
                    num_frame)).float()

        #-------- The loss come from 2D centerline projection --------
        if cal_loss_target == 'centerline_2d':

            #-------- Get the projection point for 3D simpoints --------
            projection2dpoints_3Dsim_node = LossTypeList.get2DProjectionPointfrom3DSimPoints(
                node_pos_curr)

            #-------- Get the projection loss for two 2D projection lines --------
            loss_projection_lineseg_2D = LossTypeList.getProjectionLineSegmentsLoss(
                projection2dpoints_3Dsim_node.shape[0],
                centerline_2d_image.shape[0], projection2dpoints_3Dsim_node,
                centerline_2d_image) / centerline_2d_image.shape[0]

            loss = loss_projection_lineseg_2D

        #-------- The loss comes from the lowest point on the 3D centerline --------
        if cal_loss_target == 'pt_lwst_3Dcenterline':

            #-------- Get the lowest point along the gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_PointAlongGraivity
        #pdb.set_trace()
        #-------- The loss from 3D centerline projection --------
        if cal_loss_target == 'centerline_3d_gravity':

            loss_projection_lineseg_3D = LossTypeList.getProjectionLineSegmentsLoss(
                node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                node_pos_curr, centerline_3d_gravity)

            loss = loss_projection_lineseg_3D

        #-------- The loss from 3 points on the 3D centerline (any two points + lowest point) --------
        if cal_loss_target == '3D_3pt':
            #-------- Get Point to point loss --------
            real2sim_index_SimNode = torch.as_tensor([2, -1])
            P2P_Loss = LossTypeList.getP2PLoss(centerline_3d_gravity,
                                               node_pos_curr,
                                               real2sim_index_SimNode,
                                               two_dimension=False)

            #-------- Get loss along gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            loss = P2P_Loss + loss_PointAlongGraivity

        #-------- The loss from 3 points on the 3D centerline (any two points + lowest point) --------
        if cal_loss_target == 'centerline_2d_plus_lwst':

            #-------- get the projection 2d points from 3d sim node --------
            projection2Dpoints_3Dsim_node = LossTypeList.get2DProjectionPointfrom3DSimPoints(
                node_pos_curr)

            #-------- get the 2d projection loss --------
            loss_projection_lineseg_2D = LossTypeList.getProjectionLineSegmentsLoss(
                projection2Dpoints_3Dsim_node.shape[0],
                centerline_2d_image.shape[0], projection2Dpoints_3Dsim_node,
                centerline_2d_image) / centerline_2d_image.shape[0]

            #-------- get the lowest point loss along gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            #-------- Average the 2D projection loss --------
            loss = loss_projection_lineseg_2D + loss_PointAlongGraivity * 1000
            print("loss_projection_lineseg_2D:", loss_projection_lineseg_2D)
            print("loss_PointAlongGraivity   :", loss_PointAlongGraivity)

        #-------- The loss from 2d centerline projectio + lowest point + normal vector loss --------
        if cal_loss_target == 'centerline_2d_lwst_normal_vector':

            #-------- Get the projection point for 3D simpoints --------
            projection2Dpoints_3Dsim_node = LossTypeList.get2DProjectionPointfrom3DSimPoints(
                node_pos_curr)
            loss_projection_lineseg_2D = LossTypeList.getProjectionLineSegmentsLoss(
                projection2Dpoints_3Dsim_node.shape[0],
                centerline_2d_image.shape[0], projection2Dpoints_3Dsim_node,
                centerline_2d_image) / centerline_2d_image.shape[0]

            #-------- Get the Lowest Point loss  Along Gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            #-------- Get the Normal Vector Loss --------
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_projection_lineseg_2D + loss_PointAlongGraivity * 100 + normalVector_loss * 10

        #-------- The loss from 3d centerline projection + 3D length difference --------
        if cal_loss_target == 'centerline_3d_gravity_plus_length':
            #The loss from 3d centerline projection
            loss_projection_lineseg_3D = LossTypeList.getProjectionLineSegmentsLoss(
                node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                node_pos_curr, centerline_3d_gravity)

            #-------- The loss from3D length difference --------
            loss_lenght_3dcenterline = LossTypeList.getLossForLengthBetweenSimAndReal(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_projection_lineseg_3D * 1 + loss_lenght_3dcenterline * 100

        #-------- The loss from 3d centerline projection + normal vector loss --------
        if cal_loss_target == 'centerline_3d_gravity_plus_normalvector':
            # The loss from 3d centerline projection
            loss_projection_lineseg_3D = LossTypeList.getProjectionLineSegmentsLoss(
                node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                node_pos_curr, centerline_3d_gravity)
            #Get the Normal Vector Loss
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_projection_lineseg_3D * 1 + normalVector_loss * 10

        #else:
        #    print("The loss input is wrong.")

        return loss

    ## ================================================================================
    #   The function to calcualte the loss for correspnding
    ## ================================================================================
    def getCorrespondanceLoss(self, cal_loss_target, node_pos_curr, num_frame,
                              cam_intrinsic_mat):

        #-------- load all loss type --------
        LossTypeList = LossList(cam_intrinsic_mat)

        #-------- Load the centerline --------
        centerline_3d_gravity = torch.as_tensor(
            np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)).float()
        centerline_2d_image = torch.as_tensor(
            np.load("../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy" %
                    num_frame)).float()

        #-------- 2D center line loss --------
        if cal_loss_target == 'centerline_2d':
            loss_correspondance_lineseg_3D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            loss = loss_correspondance_lineseg_3D

        #-------- 3D center line loss --------
        if cal_loss_target == 'centerline_3d_gravity':
            loss_correspondance_lineseg_3D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            loss = loss_correspondance_lineseg_3D

        #-------- 2D center line loss + 3D lowest point --------
        if cal_loss_target == 'centerline_2d_plus3dlwst':
            loss_correspondance_lineseg_2D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_2d_image, node_pos_curr, two_dimension=True)

            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_correspondance_lineseg_2D / centerline_2d_image.shape[
                0] + loss_PointAlongGraivity * 100

        #-------- 2D center line loss + 3D lowest point +_ NormalVector loss --------
        if cal_loss_target == 'centerline_2d_plus3dlwst_plusNormalVector':
            #-------- Get the 2D projection Loss --------
            loss_correspondance_lineseg_2D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_2d_image, node_pos_curr, two_dimension=True)

            #-------- Get the Lowest Point loss  Along Gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            #-------- Get the Normal Vector Loss --------
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_correspondance_lineseg_2D + loss_PointAlongGraivity * 100 + normalVector_loss * 10

        #-------- 2D center line loss + 3D lowest point +_ NormalVector loss --------
        if cal_loss_target == 'centerline_2d_pluslengthfrom3dcenterline':
            loss_correspondance_lineseg_3D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            loss_lenght_3dcenterline = LossTypeList.getLossForLengthBetweenSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_correspondance_lineseg_3D * 1 + loss_lenght_3dcenterline * 100

        #-------- 2D center line loss +_ NormalVector loss --------
        if cal_loss_target == 'centerline_2d_plusnormalvector':
            loss_correspondance_lineseg_3D, _ = LossTypeList.getP2PCorrespondLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            #-------- Get the Normal Vector Loss --------
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_correspondance_lineseg_3D * 1 + normalVector_loss * 10

        return loss

    def getSegmentsProjectionLoss(self, cal_loss_target, node_pos_curr,
                                  num_frame, cam_intrinsic_mat):

        #-------- load all loss type --------
        LossTypeList = LossList(cam_intrinsic_mat)

        #-------- Load the centerline --------
        centerline_3d_gravity = torch.as_tensor(
            np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)).float()
        centerline_2d_image = torch.as_tensor(
            np.load("../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy" %
                    num_frame)).float()

        #-------- 2D centerline loss --------
        if cal_loss_target == 'centerline_2d':
            loss_segpro_line_2D = LossTypeList.getSegProjectionLineLoss(
                centerline_2d_image, node_pos_curr, two_dimension=True)
            #Average the 2D projection loss
            loss = loss_segpro_line_2D / centerline_2d_image.shape[0]
        #pdb.set_trace()
        #-------- 3D centerline loss --------
        if cal_loss_target == 'centerline_3d_gravity':
            loss_segpro_line_3D = LossTypeList.getSegProjectionLineLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            loss = loss_segpro_line_3D

        #--------  2D centerline loss + 3D lowest point --------
        if cal_loss_target == 'centerline_2d_plus_lwst':
            loss_segpro_line_2D = LossTypeList.getSegProjectionLineLoss(
                centerline_2d_image, node_pos_curr,
                two_dimension=True) / centerline_2d_image.shape[0]

            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            loss = loss_segpro_line_2D + loss_PointAlongGraivity * 100

        #-------- 2D centerline loss + 3D lowest point + normal vector --------
        if cal_loss_target == 'centerline_2d_plus3dlwst_plusNormalVector':
            ##Get the 2D projection Loss
            loss_segpro_line_2D = LossTypeList.getSegProjectionLineLoss(
                centerline_2d_image, node_pos_curr,
                two_dimension=True) / centerline_2d_image.shape[0]

            #-------- Get the Lowest Point loss  Along Gravity --------
            loss_PointAlongGraivity, _, _ = LossTypeList.getLowestPointAlongGravityLoss(
                centerline_3d_gravity, node_pos_curr)

            #-------- Get the Normal Vector Loss --------
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_segpro_line_2D + loss_PointAlongGraivity * 300 + normalVector_loss * 10

        #-------- 2D centerline loss + 3D lowest point + normal vector --------
        if cal_loss_target == 'centerline_2d_pluslengthfrom3dcenterline':
            loss_segpro_line_2D = LossTypeList.getSegProjectionLineLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            loss_lenght_3dcenterline = LossTypeList.getLossForLengthBetweenSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_segpro_line_2D * 1 + loss_lenght_3dcenterline * 100

        #-------- 2D centerline loss  + normal vector --------
        if cal_loss_target == 'centerline_2d_plusnormalvector':
            loss_segpro_line_2D = LossTypeList.getSegProjectionLineLoss(
                centerline_3d_gravity, node_pos_curr, two_dimension=False)
            #-------- Get the Normal Vector Loss --------
            normalVector_loss = LossTypeList.getNormalVectorLossforSimAndReal(
                centerline_3d_gravity, node_pos_curr)
            loss = loss_segpro_line_2D * 1 + normalVector_loss * 10
        return loss


## ================================================================================
#   Execute the Backward Process
## ================================================================================


class ExecuteBackward():
    def Execute(self, start_frame, end_frame, save_dir, LR_RATE, LAYERS, STEPS,
                ITERATION, use_2D_centerline, backward, cal_loss_method,
                cal_loss_target):
        ## ================================================================================
        #    Initial the prerequired data
        ## ================================================================================
        BaxterRope = PBDRope(start_frame, end_frame, save_dir, LR_RATE, LAYERS,
                             STEPS, ITERATION)
        #pdb.set_trace()

        # -------- dist_velocity = torch.zeros_like(BaxterRope.node_pos_curr) --------
        dist_gravity = torch.zeros_like(BaxterRope.node_pos_curr)

        ## ================================================================================
        #    Initial all the constraints
        ## ================================================================================

        # -------- Inital the Distance constraint --------
        Diff_dist = AutoDiffDistance(BaxterRope.goal_dist,
                                     BaxterRope.compliance, BaxterRope.dt)

        # -------- Inital the stretch constraint --------
        Diff_stretch = AutoDiffStretch(BaxterRope.compliance, BaxterRope.dt,
                                       BaxterRope.goal_dist)

        #-------- Inital the bending constraint --------
        Diff_bending = AutoDiffBending(BaxterRope.compliance, BaxterRope.dt,
                                       BaxterRope.DarbouxVector)

        # #-------- Inital the vol1 constraint --------
        # Diff_vol1 = PBDRope.AutoDiffSimVolume(BaxterRope.compliance,
        #                                       BaxterRope.dt,
        #                                       BaxterRope.goal_volume1)

        # #-------- Inital the vol2 constraint --------
        # Diff_vol2 = PBDRope.AutoDiffComVolume(BaxterRope.compliance,
        #                                       BaxterRope.dt,
        #                                       BaxterRope.goal_volume2)
        # #-------- Inital the SimShape constraint --------
        # Diff_shape1 = PBDRope.AutoSimShape(BaxterRope.xp, BaxterRope.ri,
        #                                    BaxterRope.node_num)

        # #-------- Inital the ComShape constraint --------
        # Diff_shape2 = PBDRope.AutoComShape(BaxterRope.xp,
        #                                    BaxterRope.node_radius_goal,
        #                                    BaxterRope.node_quaternion_goal,
        #                                    BaxterRope.node_num - 1)

        # #-------- Inital the radius constraint --------
        # Diff_radius = PBDRope.AutoDiffRadius(BaxterRope.compliance,
        #                                      BaxterRope.dt,
        #                                      BaxterRope.node_radius_goal)

        # #-------- Initial the gravity constraint --------
        # Diff_gravity = PBDRope.AutoDiffGravity()

        # delta_x = []

        goal_pos = np.load('../DATA/goal_traj.npy')

        for num_frame in range(BaxterRope.start_frame, BaxterRope.end_frame):

            ## -------------------------------------
            # Load the centerline Data
            ## -------------------------------------

            centerline_3d_gravity = np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)
            centerline_3d_gravity = torch.as_tensor(
                centerline_3d_gravity).float() * 1.0
            if use_2D_centerline == True:
                centerline_2d_image = np.load(
                    "../DATA/DATA_BaxterRope/centerline_2d_image/%d.npy" %
                    num_frame)
                centerline_2d_image = torch.as_tensor(
                    centerline_2d_image) * 1.0

            # ---------- Load the previous frame's sim_node -----------
            if num_frame != 16:

                BaxterRope.node_pos_curr = torch.as_tensor(
                    np.load(BaxterRope.save_dir +
                            "/frame_{}.npy".format(num_frame -
                                                   1))[:, :3]).float()

            #--------- Initial the initial loacation from previous frame which can reduce the computation time ---------
            BaxterRope.update_pos_actuated = BaxterRope.node_pos_curr[
                BaxterRope.id_node_actuated, :].clone()  #Add some noise
            BaxterRope.update_pos_actuated = Variable(
                BaxterRope.update_pos_actuated, requires_grad=True)
            BaxterRope.optimizer = optim.Adam([BaxterRope.update_pos_actuated],
                                              lr=BaxterRope.LR_RATE,
                                              weight_decay=0.000)

            ## -------------------------------------
            # initialize data analysis history
            ## -------------------------------------

            save_loss_itr = []
            save_curr_itr_time = []
            save_total_itr_time = []
            save_nodes_pos_itr = []
            save_nodes_radius_itr = []
            save_nodes_quaternion_itr = []
            C_dis_itrs = []
            C_bending_itrs = []
            C_strain_itrs = []

            time_begin_frame = time.time()
            BaxterRope.optimizer.zero_grad()

            for itr in range(BaxterRope.ITERATION):

                time_begin_itr = time.time()

                # reset optimizer gradients
                BaxterRope.optimizer.zero_grad()

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
                BaxterRope.node_quaternion_curr = torch.div(
                    BaxterRope.node_quaternion_curr.T,
                    torch.linalg.norm(BaxterRope.node_quaternion_curr,
                                      dim=1)).T
                BaxterRope.node_quaternion_curr = torch.nan_to_num(
                    BaxterRope.node_quaternion_curr, nan=0)

                ## ================================================================================
                #     Initial constraints parameters
                ## ================================================================================
                Lambda_strain = torch.zeros((BaxterRope.node_num - 1, 3))
                # Lambda_vol1 = torch.zeros(BaxterRope.node_num)
                # Lambda_radius = torch.zeros(BaxterRope.node_num)
                # Lambda_shape1 = 0

                Lambda_dist = torch.zeros((BaxterRope.node_num - 1, 3))
                Lambda_bending = torch.zeros((BaxterRope.node_num - 2, 3))
                # constraint_q1 = Quaternion([1, 0, 0, 0]).normalised
                # constraint_q2 = Quaternion([1, 0, 0, 0]).normalised

                node_pos_curr = BaxterRope.node_pos_curr.detach()
                node_radius_curr = BaxterRope.node_radius_curr.detach()
                node_quaternion_curr = BaxterRope.node_quaternion_curr.detach()

                ## Set actuation point
                # node_pos_curr[BaxterRope.id_node_actuated, :] = BaxterRope.update_pos_actuated
                if backward == False:
                    node_pos_curr[
                        BaxterRope.
                        id_node_actuated, :] = centerline_3d_gravity[0]
                else:
                    node_pos_curr[
                        BaxterRope.
                        id_node_actuated, :] = BaxterRope.update_pos_actuated
                #pdb.set_trace()
                # ### ==========================================================================================
                # ## Plotting
                OptimizedData = ProcessOptData()
                # OptimizedData.ShowImage(BaxterRope.ITERATION,
                #                         centerline_3d_gravity, node_pos_curr,
                #                         save_loss_itr, save_C_dis_max_itr,
                #                         save_C_bending_max_itr,
                #                     save_C_strain_max_itr)
                ## ================================================================================
                #    Main Loops
                ## ================================================================================
                for j in range(BaxterRope.STEPS):

                    ### ------- Gravity constraints -------
                    # dist_gravity[1:-1, :] = 9.8 * (BaxterRope.dt**
                    #                                2) * 0.5 * torch.as_tensor(
                    #                                    BaxterRope.gravity_dir) * wg

                    # gravity_stiffness, sim_plane_normal_vec_angle = Diff_gravity.ifSimPlaneInGravityDir(
                    #     centerline_3d_gravity, node_pos_curr, BaxterRope.gravity_dir)
                    # dist_gravity[1:-1, :] = 9.8 * (BaxterRope.dt**
                    #                                2) * 0.5 * BaxterRope.gravity_dir * wg * 1 * gravity_stiffness
                    dist_gravity = 9.8 * (
                        BaxterRope.dt**2
                    ) * 0.5 * BaxterRope.gravity_dir * BaxterRope.wg * 1 * 1.0
                    # node_pos_curr = node_pos_curr + (dx_dist * w_dis + dist_gravity * w_gravity) / (w_gravity + w_dis)

                    dist_gravity_repeat = dist_gravity.repeat(
                        BaxterRope.node_num, 1)
                    dist_gravity_repeat[0] = 0 * dist_gravity_repeat[0]
                    dist_gravity_repeat[-1] = 0 * dist_gravity_repeat[-1]
                    #pdb.set_trace()
                    node_pos_curr = node_pos_curr + (
                        dist_gravity_repeat * abs(BaxterRope.w_gravity)) / 1
                    # node_pos_curr = node_pos_curr + dist_gravity * w_gravity
                    # node_pos_curr = node_pos_curr + dx_dist * w_dis
                    # pdb.set_trace()

                    for i in range(BaxterRope.LAYERS):
                        #pdb.set_trace()

                        ### ------- TDMAL distance constraints -------
                        dx_dist, Lambda_dist, C_dis = Diff_dist.solveDistConstraintTDMALinearSolver(
                            node_pos_curr, Lambda_dist, BaxterRope.wab,
                            BaxterRope.wab)
                        #pdb.set_trace()

                        # node_pos_curr = node_pos_curr + dx_dist * w_dis
                        #pdb.set_trace()

                        ### ------- Stretch constraint -------
                        dx_strain, Lambda_strain, dq_strain, C_strain = Diff_stretch.C_strain_solve(
                            node_pos_curr, Lambda_strain, node_quaternion_curr,
                            BaxterRope.ws, BaxterRope.ws, BaxterRope.wq)
                        dx_strain[0] = dx_strain[0] * 0
                        dx_strain[-1] = dx_strain[-1] * 0

                        # node_pos_curr = node_pos_curr + dx_strain * w_strain
                        # node_pos_curr = node_pos_curr + (dx_strain * w_strain + dx_dist * w_dis) / 2

                        ### ------- Bending constraint -------
                        dq_bending, Lambda_bending, C_bending = Diff_bending.solve_BendTwistConstraint(
                            node_pos_curr, node_quaternion_curr,
                            Lambda_bending, BaxterRope.wq, BaxterRope.wq)

                        ### ------- Radius constraint -------
                        # dr_radius, Lambda_radius = Diff_radius.C_radius_solve(Lambda_radius, node_radius_curr)

                        ### ------- Distance constraint -------
                        # dx_dist, Lambda_dist = Diff_dist.solveDistConstraint(node_pos_curr, Lambda_dist, wa * scale_wab,
                        #                                                      wb * scale_wab)
                        # dx_dist, Lambda_dist = Diff_dist.solveDistConstraintTDMALinearSolver(
                        #     node_pos_curr, Lambda_dist, wa * scale_wab, wb * scale_wab)

                        ### ------- Volume constraint -------
                        # dx_vol1, dr_vol1, Lambda_vol1 = Diff_vol1.solveVolumeConstraint(node_pos_curr, node_radius_curr,
                        #                                                                 Lambda_vol1)

                        ### ------- Volume2 constraint -------
                        # dx_vol2, dr_vol2, Lambda = Diff_vol2.solveVolumeConstraint(node_pos_curr, node_radius_curr)

                        ### ------- Shape matching constraint -------
                        # dx_shapesim, constraint_q1, L = Diff_shape1.solveShapeMatchingConstraint(
                        #     node_pos_curr, node_radius_curr, constraint_q1, L)
                        # dx_com, dr_com, dq_com, constraint_q2 = Diff_shape2.solveShapeMatchingConstraint(
                        #     node_pos_curr, node_quaternion_curr, node_radius_curr, constraint_q2)

                        # pdb.set_trace()

                        ### ==========================================================================================
                        ###       Updating states
                        ### ==========================================================================================
                        ### position
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_dist + dx_strain * 6.0 + dist_gravity) / 8.0 * 1
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_strain * 6.0 + dist_gravity) / 7.0 * 1
                        # node_pos_curr = node_pos_curr + 1.00 * (dx_strain * w_strain) / 1.0
                        # node_pos_curr = node_pos_curr + dx_strain * w_strain
                        node_pos_curr = node_pos_curr + (
                            dx_strain * BaxterRope.w_strain +
                            dx_dist * BaxterRope.w_dis) / 2 * BaxterRope.w_SOR
                        # node_pos_curr = node_pos_curr + (dx_strain * w_strain + dx_dist * w_dis) / (w_strain + w_dis)

                        ## quaternion
                        node_quaternion_curr = node_quaternion_curr + (
                            dq_strain * BaxterRope.wq_strain + dq_bending *
                            BaxterRope.wq_bending) / 2 * BaxterRope.w_SOR
                        # node_quaternion_curr = node_quaternion_curr + (dq_strain * wq_strain +
                        #                                                dq_bending * wq_bending) / (wq_strain + wq_bending)

                        node_quaternion_curr = torch.div(
                            node_quaternion_curr.T,
                            torch.linalg.norm(node_quaternion_curr, dim=1) +
                            1e-7).T
                        #node_quaternion_curr = torch.nan_to_num(
                        #     node_quaternion_curr, nan=0)
                        # # ---------- Save Temp DATA -------
                        # DiffLoss = GetLoss()
                        # loss_temp = DiffLoss.CalculateLoss(
                        #     cal_loss_method, cal_loss_target, node_pos_curr,
                        #     num_frame, BaxterRope.cam_intrinsic_mat)
                        # #pdb.set_trace()

                        # OptimizedData.SaveTempData(loss_temp,
                        #                            C_dis.detach().numpy(),
                        #                            C_strain.detach().numpy(),
                        #                            C_bending.detach().numpy(),
                        #                            itr, j, i, cal_loss_method,
                        #                            cal_loss_target)

                ### ==========================================================================================
                ###       Updating BaxterRope related information
                ### ==========================================================================================

                BaxterRope.node_pos_curr = node_pos_curr
                BaxterRope.node_quaternion_curr = node_quaternion_curr
                BaxterRope.node_radius_curr = node_radius_curr

                ### ==========================================================================================
                ##         Get different loss
                ### ==========================================================================================
                # loss_projection_lineseg_3D = BaxterRope.getProjectionLineSegmentsLoss(
                #     node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                #     node_pos_curr, centerline_3d_gravity)

                loss = torch.linalg.norm(node_pos_curr[0] -
                                         goal_pos[num_frame])

                ### ==========================================================================================
                ##         print related backward result
                ### ==========================================================================================

                time_finish_frame = time.time()
                curr_itr_time = time_finish_frame - time_begin_itr
                total_itr_time = time_finish_frame - time_begin_frame
                print("\n")
                print("itr", itr)
                print("Curr  itr time :", curr_itr_time)
                print("Total itr time :", total_itr_time)
                print("Loss           :", loss.detach().numpy())
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

                C_dis_itr = torch.abs(C_dis)
                C_bending_itr = torch.linalg.norm(C_bending, dim=1)
                C_strain_itr = torch.linalg.norm(C_strain, dim=1)
                #pdb.set_trace()

                ### ==========================================================================================
                ### Do loss backpropagation
                # loss.backward(retain_graph=True)
                if backward == True:
                    loss.backward()
                    BaxterRope.optimizer.step()

                print("The gradient of control point:",
                      BaxterRope.update_pos_actuated.grad)
                print("The position of control point:",
                      BaxterRope.update_pos_actuated.detach().numpy())
                print("The ground true control point:",
                      centerline_3d_gravity[0].detach().numpy())

                ### ==========================================================================================
                ### Recording
                # ------- Save time and loss information -------
                save_loss_itr.append(loss.detach().numpy())
                save_curr_itr_time.append(curr_itr_time)
                save_total_itr_time.append(total_itr_time)

                # ------- Save final result information -------
                save_nodes_pos_itr.append(
                    BaxterRope.node_pos_curr.detach().numpy())
                save_nodes_radius_itr.append(
                    BaxterRope.node_radius_curr.detach().numpy())
                save_nodes_quaternion_itr.append(
                    BaxterRope.node_quaternion_curr.detach().numpy())

                # ------- Save Constraint information -------
                C_dis_itrs.append(C_dis_itr.detach().numpy())
                C_bending_itrs.append(C_bending_itr.detach().numpy())
                C_strain_itrs.append(C_strain_itr.detach().numpy())

            ### ==========================================================================================
            ### Processingv optimization Data
            OptimizedData = ProcessOptData()

            # # ### ==========================================================================================
            # ## Plotting
            # OptimizedData.ShowImage(BaxterRope.ITERATION,
            #                         centerline_3d_gravity, node_pos_curr,
            #                         save_loss_itr, C_strain_itr, C_bending_itr,
            #                         C_dis_itr)

            #plt.pause(1.0)
            #pdb.set_trace()
            # ==========================================================================================
            ## Save Data
            OptimizedData.SaveData(save_loss_itr, save_curr_itr_time,
                                   save_total_itr_time, save_nodes_pos_itr,
                                   save_nodes_radius_itr,
                                   save_nodes_quaternion_itr, C_dis_itrs,
                                   C_bending_itrs, C_strain_itrs,
                                   BaxterRope.node_num, BaxterRope.save_dir,
                                   num_frame)
