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


class PBDRope():
    def __init__(self):
        # ---------- define number of particles ----------
        self.node_num = 20
        self.compliance = 2e-6
        # self.compliance = 0.002
        self.dt = 0.1
        self.damper = 0.0

        # ---------- Frame Information ----------
        self.initial_frame = 14
        self.start_frame = 30
        self.end_frame = 162

        # ---------- prefined information ----------
        self.cam_intrinsic_mat = torch.as_tensor(
            [[960.41357421875, 0.0, 1021.7171020507812],
             [0.0, 960.22314453125, 776.2381591796875], [0.0, 0.0, 1.0]])
        gravity_dir = torch.as_tensor([0.10381715, 0.9815079, -0.16082364])
        self.gravity_dir = gravity_dir / torch.linalg.norm(gravity_dir, dim=0)
        self.save_dir = '../save_loss_withConstConvergency/centerline_3d_gravity_p2p/'

        ### ------- Constraints stiffness -------
        self.w_gravity = 0.20070688

        self.w_dis = 0.87617564

        self.w_strain = 1.1042638  ## effect : if on the same plane

        self.wq_strain = 0.8733262

        self.wq_bending = 1.0066655  ## effect : curvature

        self.w_SOR = 1.3067461

        #------------ Create Variable ------------
        self.w_gravity = Variable(torch.as_tensor(self.w_gravity),
                                  requires_grad=True)
        self.w_dis = Variable(torch.as_tensor(self.w_dis), requires_grad=True)
        self.w_strain = Variable(torch.as_tensor(self.w_strain),
                                 requires_grad=True)

        self.wq_strain = Variable(torch.as_tensor(self.wq_strain),
                                  requires_grad=True)
        self.wq_bending = Variable(torch.as_tensor(self.wq_bending),
                                   requires_grad=True)
        self.w_SOR = Variable(torch.as_tensor(self.w_SOR), requires_grad=True)

        #------------  set gradient optimizer ------------
        self.LR_RATE = 1e-2
        self.LAYERS = 1 * 5
        self.STEPS = 50
        self.ITERATION_OPT = 100
        self.backward = True
        self.use_2D_centerline = True

        #------------ set goals ------------
        self.setGoalState()

        #------------ initialization ------------
        self.setInitialization()

        #------------ set actuated/target ID ------------
        id_node_target = []
        id_node_actuated = []
        id_node_fixed = []

        for i in range(self.node_num):
            if i != 0:
                id_node_target.append(i)
            # if i%2==0:
            elif i == 0:
                id_node_actuated.append(i)

        self.id_node_target = torch.as_tensor(id_node_target)
        self.id_node_actuated = torch.as_tensor(id_node_actuated)
        self.id_node_fixed = torch.as_tensor(id_node_fixed)

        # --------- set control point and optimizer --------
        self.update_pos_actuated = Variable(self.node_pos_curr[0],
                                            requires_grad=True)

        # self.update_pos_actuated = Variable(self.node_pos_curr[], requires_grad=True)
        # update_radius = Variable(radius[id_actuated_points], requires_grad=True)
        # self.node_pos_curr[self.id_node_actuated] = self.update_pos_actuated[0]
        # self.optimizer = optim.Adam([self.update_pos_actuated],
        #                             lr=self.LR_RATE,
        #                             weight_decay=0.000)

        self.optimizer = optim.Adam([
            self.w_gravity, self.w_dis, self.w_strain, self.wq_strain,
            self.wq_bending, self.w_SOR
        ],
                                    lr=self.LR_RATE,
                                    weight_decay=0.000)
        self.loss = None
        torch.autograd.set_detect_anomaly(True)

    def setGoalState(self):

        # --------- define goal position of particle ---------
        self.node_pos_goal = torch.zeros((self.node_num, 3))
        centerline_3d_gravity = np.load(
            "../DATA_BaxterRope/downsampled_pcl/initialization.npy") * 1.0
        min = centerline_3d_gravity[0]
        max = centerline_3d_gravity[-1]

        distance = (max - min) / (self.node_num - 1)
        for i in range(self.node_num):
            self.node_pos_goal[i] = torch.as_tensor(min + i * distance) * 1.0

        # --------- define goal radius of particles ----------
        self.node_radius_goal = torch.ones(self.node_num)
        for i in range((self.node_num)):
            self.node_radius_goal[i] = 0.01

        # ---------- get Goal rest lenght ----------
        self.goal_dist = self.RestLength(self.node_pos_goal)

        # ----------- get goal volume for simple one ------------
        self.goal_volume1 = self.VolumeSim(self.node_pos_goal,
                                           self.node_radius_goal)

        # ---------- get goal volume for complex one ------------
        self.goal_volume2 = self.VolumeCom(self.node_pos_goal,
                                           self.node_radius_goal)

        # ------------ Get the quaternion,rotation matrix and restlength for the goal pos ------------
        self.node_rotation, self.node_length_goal, self.node_quaternion_goal = self.Quaternion(
            self.node_pos_goal)

        # ------------ Get the rest shape for the complex shape matching ------------
        self.center, self.xp, self.ri = self.SimShape(self.node_pos_goal,
                                                      self.node_radius_goal)

        # ------------ Get the DarbouxVector for the bending constraint ------------
        self.DarbouxVector = self.DarbouxRest(self.node_pos_goal,
                                              self.node_quaternion_goal)
        # self.node_pos_goal[15, 0] += 40

    def setInitialization(self):

        # --------- define curr position of particle ---------
        initial_states = torch.as_tensor(
            np.load(self.save_dir + "frame_{}_init.npy".format(
                self.initial_frame))).float() * 1
        self.node_pos_curr = initial_states[:, 0:3]

        # --------- define curr radius of particle ---------
        self.node_radius_curr = initial_states[:, 3]
        # self.node_radius_curr = self.node_radius_goal + r_offset

        # --------- define curr quaternion of particle ---------
        _, _, self.node_quaternion_curr = self.Quaternion(self.node_pos_curr)

    # --------- Rest Length ---------
    def RestLength(self, pos):
        diff_pos = torch.diff(pos, dim=0).float()
        diff_pos = torch.linalg.norm(diff_pos, dim=1)
        return diff_pos

    # --------- Rest Volume for Simple volume constraint ---------
    def VolumeSim(self, pos, radius):
        radius_a = radius[0:-1]
        radius_b = radius[1:]

        diff_pos = torch.diff(pos, dim=0).float()
        dis_pos = torch.linalg.norm(diff_pos, ord=None, dim=1)

        r0 = 0.5 * (radius_a + radius_b)

        vol = r0 * r0 * dis_pos
        return vol

    # --------- Rest Volume for complex volume constraint ---------
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

    # --------- Get quaternion ---------
    def Quaternion(self, pos):
        """
        Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        node_num = self.node_num
        vec1 = np.array([0, 0, 1])
        q = torch.zeros((node_num - 1, 4))
        rest_length = torch.ones((node_num - 1, 1))
        rot = torch.zeros((node_num - 1, 3, 3))
        # a and b are in the form of numpy array
        for i in range(pos.shape[0] - 1):
            a, b = pos[i, :3], pos[i + 1, :3]
            ab = b - a

            b_vec = torch.as_tensor(b - a) * 1.0
            b_vec = (b_vec / torch.linalg.norm(torch.FloatTensor(b_vec)))
            vec2 = b_vec.detach().numpy()
            a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
                vec2 / np.linalg.norm(vec2)).reshape(3)
            v = np.cross(a, b)
            c = np.dot(a, b)
            s = np.linalg.norm(v)
            kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]],
                             [-v[1], v[0], 0]])
            rotation_matrix = np.eye(3) + kmat + \
                kmat.dot(kmat) * ((1 - c) / (s ** 2))
            r = R.from_matrix(rotation_matrix)
            rot[i] = torch.as_tensor(r.as_matrix())
            # kk=r.as_euler('xyz', degrees=True)
            qc = r.as_quat()  # x,y,z,w'

            q[i] = torch.tensor(qc)
            rest_length[i] = torch.linalg.norm(torch.as_tensor(b - a) * 1.0)
        return rot, rest_length, q

    # ----------- For the precondition of simple shape matching ----------
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

    # ---------- Get the darboxRest for the bending constraint ----------
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

    def mulQuaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p_scalar, q_scalar = p[:, 3], q[:, 3]
        p_imag, q_imag = p[:, :3], q[:, :3]
        quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
        quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
            -1, 1) * p_imag + torch.cross(p_imag, q_imag)
        return quat_imag, quat_scalar

    def findClosestPointOnLine(self, sim_nodes, points_cloud, sim_num, gt_num):
        # ap = points_cloud - sim_nodes[:-1]

        # result = sim_nodes[:-1] +(torch.sum(ap*ab,dim=2)/(torch.sum(ab*ab,dim=1)[None,:]))[:,:,None] * ab[None,:,:]

        # l2 = torch.sum((ab)**2, dim=1)

        # if you need the point to project on line extention connecting between points
        # ab_extension = ab.repeat(gt_num, 1).reshape(gt_num, -1, sim_nodes.shape[1])
        # ab_extension = ab.tile(gt_num, 1, 1)  ## size : num_gt*(sim_num-1)*3

        # t = torch.sum((points_cloud - sim_nodes[:-1]) * ab_extension, dim=2)  ## this will be the same by multiple ab

        ab = torch.diff(sim_nodes, dim=0)
        ab_length_square = torch.linalg.norm(ab, dim=1)**2
        t = torch.sum((points_cloud - sim_nodes[:-1]) * ab, dim=2)
        t = t / ab_length_square

        negative_index = torch.where(t < 0)
        positive_index = torch.where(t > 1)
        t[negative_index] = 0
        t[positive_index] = 1

        projection = sim_nodes[:-1] + t[:, :, None] * ab[None, :, :]

        # ab_extension = ab.repeat(gt_num, 1).reshape(gt_num, sim_num - 1, sim_nodes.shape[1])

        # projection = sim_nodes[:-1] + t[:, :, None] * ab_extension

        return projection

    def getProjectionLineSegmentsLoss(self, sim_num, gt_num, sim_nodes,
                                      gt_points):

        # sim_nodes=torch.ones((30,3))*20
        # gt_points=torch.ones((300,3))*20

        # point_line = torch.full((gt_num, 29), float("Inf"))
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

        # pdb.set_trace()

        return torch.sum(dmin)

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

        # pdb.set_trace()

        return loss_correpond, real2sim_id.indices

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

    def getLowestPointAlongGravityLoss(self, centerline_3d_gravity,
                                       node_pos_curr):

        # point_line = torch.full((gt_num, 29), float("Inf"))

        # line_gravity = np.vstack((centerline_3d_gravity[0].detach().numpy(),
        #                           (centerline_3d_gravity[0] + self.gravity_dir * 0.5).detach().numpy()))
        # ax = plt.axes(projection='3d')
        # ax.plot(centerline_3d_gravity[:, 0].detach().numpy(),
        #         centerline_3d_gravity[:, 1].detach().numpy(),
        #         centerline_3d_gravity[:, 2].detach().numpy(),
        #         color='#B61919',
        #         linewidth=2)
        # # ax.scatter(centerline_3d_gravity[:, 0],
        # #            centerline_3d_gravity[:, 1],
        # #            centerline_3d_gravity[:, 2],
        # #            c='#B61919',
        # #            s=2)
        # ax.plot(line_gravity[:, 0], line_gravity[:, 1], line_gravity[:, 2])
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.set_title('XYZ centerline_3d_gravity')
        # plt.show()

        extend_gravity_scale = 1
        gravity = torch.vstack((centerline_3d_gravity[0] -
                                self.gravity_dir * extend_gravity_scale,
                                centerline_3d_gravity[0] +
                                self.gravity_dir * extend_gravity_scale))
        gravity_vector = gravity[1] - gravity[0]
        gravity_length_square = torch.linalg.norm(gravity_vector)**2

        real_proj_gravity = torch.sum(
            (centerline_3d_gravity - gravity[0]) * gravity_vector, dim=1)
        real_proj_gravity = real_proj_gravity / gravity_length_square
        real_lowest_gravity_id = torch.max(real_proj_gravity, dim=0)
        real_lowest_gravity = centerline_3d_gravity[
            real_lowest_gravity_id.indices]

        sim_proj_gravity = torch.sum(
            (node_pos_curr - gravity[0]) * gravity_vector, dim=1)
        sim_proj_gravity = sim_proj_gravity / gravity_length_square
        sim_lowest_gravity_id = torch.max(sim_proj_gravity, dim=0)
        sim_lowest_gravity = node_pos_curr[sim_lowest_gravity_id.indices]

        loss_lowest_point_gravity = torch.linalg.norm(real_lowest_gravity -
                                                      sim_lowest_gravity)

        # pdb.set_trace()

        return loss_lowest_point_gravity

    def getProjectionPoint_from3Dsimpoints(self, sim_nodes):
        sim_nodes_3d_project = torch.matmul(self.cam_intrinsic_mat,
                                            sim_nodes.float().T).T
        sim_nodes_2d_normalized = torch.div(
            sim_nodes_3d_project, (sim_nodes_3d_project[:, 2]).reshape(-1, 1))
        return sim_nodes_2d_normalized[:, :2]


class AutoDiffStretch():
    def __init__(self, compliance, dt, rest_length):
        self.compliance = compliance
        self.dt = dt
        self.rest_length = rest_length

    def C_strain_solve(self, pos, Lamda, quat, wa, wb, wq):

        dq = torch.zeros((len(pos) - 1, 4))
        dx = torch.zeros((len(pos), 3))

        # dq = torch.zeros((len(q)-1, 4))
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

        # for i in range(len(qc)):
        #    d3[i, 2] = z[i] * z[i] - x[i]*x[i] - y[i]*y[i] + z[i]*z[i]

        # print(d3[0])
        # a[0] = torch.as_tensor([5.968571, 3.031847, -5.871643])
        # b[0] = torch.as_tensor([5.755610, 2.515785, -5.991727])
        # self.rest_length[0] = torch.as_tensor([0.571634])

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
        dx[1:] += wb * dp
        dq = torch.hstack((dq_imag, dq_scalar.reshape(-1, 1)))  # x,y,z,w

        Lamda += dL
        # print(dp)

        return dx, Lamda, dq, Cx

    def mul_quaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p0, q0 = p[:, 3], q[:, 3]
        P, Q = p[:, :3], q[:, :3]

        pq0 = p0 * q0 - torch.sum(P * Q, dim=1)

        PQ = p0.reshape(-1, 1) * Q + q0.reshape(-1, 1) * P + torch.cross(P, Q)

        return PQ, pq0


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

        dx = torch.zeros((len(pos), 3))
        diff_pos = torch.diff(pos, dim=0).float()
        diff_pos_norm = torch.linalg.norm(diff_pos, dim=1)
        diff_pos_transpose = diff_pos.T
        const_dis = diff_pos_norm - self.rest_length[:]

        gradC_p = torch.div(-diff_pos_transpose,
                            diff_pos_norm)  ## dim : 3*(N-1)
        gradC_p_T = gradC_p.T  ## dim : (N-1)*3
        # gradC_pp1_T = -gradC_p_T

        gradCp_CpT = torch.diagonal(
            torch.matmul(gradC_p_T[1:, :], gradC_p[:, :-1]))  ## dim : (N-1)*1

        vec_b = torch.zeros_like(diff_pos_norm) - 2
        # vec_b[0] = -2
        # vec_b[-1] = -2
        vec_b[0] = -1
        vec_b[-1] = -1

        vec_a = torch.hstack((torch.zeros(1), gradCp_CpT))
        vec_c = torch.hstack((gradCp_CpT, torch.zeros(1)))
        vec_d = const_dis

        # pdb.set_trace()

        # thomas_A_mat = self.getTridiagMatrix(vec_a, vec_b, vec_c, -1, 0, 1)
        # thomas_b_vec = const_dis

        Lamda = self.solveTDMA(vec_a, vec_b, vec_c, vec_d)

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

        num_const = gradC_p.T.shape[0]
        mat_gradC_p = torch.zeros((3 * (num_const + 1), num_const))
        id_diag0 = torch.arange(num_const * 3).reshape(-1, 3)
        id_diagN1 = torch.arange(3, (num_const + 1) * 3, 1).reshape(-1, 3)
        mat_gradC_p[
            id_diag0,
            torch.arange(num_const).reshape(-1, 1).repeat(1, 3)] = gradC_p.T
        mat_gradC_p[
            id_diagN1,
            torch.arange(num_const).reshape(-1, 1).repeat(1, 3)] = -gradC_p.T

        mat_gradC_p[0:3, 0] = 0
        mat_gradC_p[-3:, -1] = 0

        dx = torch.matmul(mat_gradC_p, Lamda).reshape(-1, 3)

        # pdb.set_trace()

        return dx, Lamda, const_dis

    def getTridiagMatrix(self, a, b, c, k1=-1, k2=0, k3=1):
        return torch.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)

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


# Simple Volume Constraint


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


# Complex Volume Constraint


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
        ra3 = 0
        rb3 = 0

        rarb = 0

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


# Simple shape matching Constraint


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


# Complex shape matching constraint


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


# Radius Constraint


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


# The bending Constraint


class AutoDiffBending():
    def __init__(self, compliance, dt, darbouxRest):
        self.compliance = compliance
        self.dt = dt
        self.darbouxRest = darbouxRest

    def solve_BendTwistConstraint(self, target, quat, Lambda, wa, wb):

        dq = torch.zeros((target.shape[0] - 1, 4))
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

        # self.darbouxRest[0] = torch.as_tensor(
        #    [0.214160, -0.099520, -0.000316, 0.971716])

        omega_plus = omega + self.darbouxRest
        omega = omega - self.darbouxRest
        for i in range(omega.shape[0]):
            if torch.linalg.norm(omega[i], ord=2) > torch.linalg.norm(
                    omega_plus[i], ord=2):
                omega[i] = omega_plus[i]

        # alpha = 0.0
        alpha = self.compliance / (self.dt * self.dt)
        Cx = torch.as_tensor(omega)
        # print(lamda[i,:].shape)

        dL = (Cx + alpha * Lambda) / (wa + wb + alpha)
        # dL/=np.linalg.norm(dL)
        dLq = torch.zeros((dL.shape[0], 4))
        dLq[:, 0], dLq[:, 1], dLq[:, 2] = dL[:, 0], dL[:, 1], dL[:, 2]

        pq_image, pq_scalar = self.mul_quaternion(Q, dLq)  # w,x,y,z
        da = torch.hstack((pq_image, pq_scalar.reshape(-1, 1)))
        pq_image, pq_scalar = self.mul_quaternion(-q0, dLq)  # w,x,y,z
        db = torch.hstack((pq_image, pq_scalar.reshape(-1, 1)))

        dq[:-1] += da * wa
        dq[1:] += db * wb
        Lambda += dL

        return dq, Lambda, Cx

    def mul_quaternion(self, p, q):
        # p and q are represented as x,y,z,w
        p_scalar, q_scalar = p[:, 3], q[:, 3]
        p_imag, q_imag = p[:, :3], q[:, :3]
        quat_scalar = p_scalar * q_scalar - torch.sum(p_imag * q_imag, dim=1)
        quat_imag = p_scalar.reshape(-1, 1) * q_imag + q_scalar.reshape(
            -1, 1) * p_imag + torch.cross(p_imag, q_imag)
        return quat_imag, quat_scalar


class Gravity_Constraint():
    def __init__(self, compliance):
        self.compliance = compliance

    def ProjectionPointAlongGravity(self, centerline_3d_gravity, sim_node,
                                    gravity_dir):

        extend_gravity_scale = 1
        gravity = torch.vstack(
            (centerline_3d_gravity[0] - gravity_dir * extend_gravity_scale,
             centerline_3d_gravity[0] + gravity_dir * extend_gravity_scale))
        gravity_vector = gravity[1] - gravity[0]
        gravity_length_square = torch.linalg.norm(gravity_vector)**2

        sim_proj_gravity = torch.sum((sim_node - gravity[0]) * gravity_vector,
                                     dim=1)
        sim_proj_gravity = sim_proj_gravity / gravity_length_square
        sim_lowest_gravity_id = torch.max(sim_proj_gravity, dim=0)
        sim_lowest_gravity = sim_node[sim_lowest_gravity_id.indices]

        sim_highest_gravity_id = torch.min(sim_proj_gravity, dim=0)
        sim_highest_gravity = sim_node[sim_highest_gravity_id.indices]

        return sim_lowest_gravity, sim_highest_gravity

    def ifSimPlaneInGravityDir(self, centerline_3d_gravity, sim_node,
                               gravity_dir):
        sim_lowest_gravity, sim_highest_gravity = self.ProjectionPointAlongGravity(
            centerline_3d_gravity, sim_node, gravity_dir)
        normalvector1 = sim_lowest_gravity - sim_node[0]
        normalvector2 = sim_lowest_gravity - sim_node[-1]
        normalvector1 = normalvector1 / torch.linalg.norm(normalvector1)
        normalvector2 = normalvector2 / torch.linalg.norm(normalvector2)
        sim_norm_vec = torch.cross(normalvector1, normalvector2)

        sim_norm_vec = sim_norm_vec / torch.linalg.norm(sim_norm_vec)

        dot_product = torch.dot(sim_norm_vec, gravity_dir)
        angle = torch.arccos(dot_product)

        gravity_stiffness = abs(torch.cos(angle)) * 1

        return gravity_stiffness, angle


if __name__ == '__main__':

    # initialize and read the image
    # initialize and read the image

    BaxterRope = PBDRope()

    print("==========================================================")
    print("\n")

    # --------- Initial The Plotting ----------
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
    ax_loss.set_xlim(0, BaxterRope.ITERATION_OPT)
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
    ax_itr_C_dis_max.set_xlim(0, BaxterRope.ITERATION_OPT)
    ax_itr_C_dis_max.set_ylim(0, 0.1)
    ax_itr_C_dis_max.set_xlabel('iter (steps)')  # Add an x-label to the axes.
    ax_itr_C_dis_max.set_ylabel('max C_dis')
    ax_itr_C_dis_max.set_title('Max C_dis constraint convergency')

    ax_itr_C_bending_max = fig.add_subplot(2, 3, 5)
    line5_itr_C_bending_max, = ax_itr_C_bending_max.plot([], [],
                                                         color='#1F6ED4',
                                                         marker='o',
                                                         markersize=3,
                                                         linestyle='-',
                                                         linewidth=1)
    ax_itr_C_bending_max.set_xlim(0, BaxterRope.ITERATION_OPT)
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
    ax_itr_C_strain_max.set_xlim(0, BaxterRope.ITERATION_OPT)
    ax_itr_C_strain_max.set_ylim(0, 2)
    ax_itr_C_strain_max.set_xlabel(
        'iter (steps)')  # Add an x-label to the axes.
    ax_itr_C_strain_max.set_ylabel('max C_strain')
    ax_itr_C_strain_max.set_title('Max C_strain constraint convergency')

    ## -------------------------------------------
    # --------- Initial The Constraints ----------
    ## -------------------------------------------
    Diff_dist = AutoDiffDistance(BaxterRope.goal_dist, BaxterRope.compliance,
                                 BaxterRope.dt)
    Diff_vol1 = AutoDiffSimVolume(BaxterRope.compliance, BaxterRope.dt,
                                  BaxterRope.goal_volume1)
    Diff_vol2 = AutoDiffComVolume(BaxterRope.compliance, BaxterRope.dt,
                                  BaxterRope.goal_volume2)
    Diff_shape1 = AutoSimShape(BaxterRope.xp, BaxterRope.ri,
                               BaxterRope.node_num)

    Diff_shape2 = AutoComShape(BaxterRope.xp, BaxterRope.node_radius_goal,
                               BaxterRope.node_quaternion_goal,
                               BaxterRope.node_num - 1)
    # print(Diff_data.node_quaternion_goal)
    Diff_radius = AutoDiffRadius(BaxterRope.compliance, BaxterRope.dt,
                                 BaxterRope.node_radius_goal)
    Diff_stretch = AutoDiffStretch(BaxterRope.compliance, BaxterRope.dt,
                                   BaxterRope.goal_dist)
    Diff_bending = AutoDiffBending(BaxterRope.compliance, BaxterRope.dt,
                                   BaxterRope.DarbouxVector)
    Diff_gravity = Gravity_Constraint(BaxterRope.compliance)

    ## -------------------------------------
    # initialize data analysis history
    ## -------------------------------------
    delta_x = []
    save_loss_itr = []
    save_curr_itr_time = []
    save_total_itr_time = []
    save_nodes_pos_itr = []
    save_nodes_radius_itr = []

    save_C_dis_MAX_itr = []
    save_C_bending_MAX_itr = []
    save_C_strain_MAX_itr = []

    save_C_dis_itr = []
    save_C_bending_itr = []
    save_C_strain_itr = []

    # -------- Save weight --------
    save_w_gravity_itr = []
    save_w_dis_itr = []
    save_w_strain_itr = []
    save_wq_strain_itr = []
    save_wq_bending_itr = []
    save_w_SOR_itr = []

    time_begin_frame = time.time()
    for itr in range(BaxterRope.ITERATION_OPT):
        loss_2frames = 0

        # if num_frame != BaxterRope.start_frame:
        #     BaxterRope.node_pos_curr = torch.as_tensor(
        #         np.load(BaxterRope.save_dir +
        #                 "frame_{}.npy".format(num_frame - 1))[:, :3]).float()

        dist_velocity = torch.zeros_like(BaxterRope.node_pos_curr)

        time_begin_itr = time.time()

        #-------- reset optimizer gradients --------
        BaxterRope.optimizer.zero_grad()
        node_pos_curr = BaxterRope.node_pos_curr.detach()
        node_radius_curr = BaxterRope.node_radius_curr.detach()
        node_quaternion_curr = BaxterRope.node_quaternion_curr.detach()

        dist_gravity = torch.zeros_like(BaxterRope.node_pos_curr).detach()

        #-------- Every optimiaztion  iteration used two frames --------
        for num_frame in [30, 35]:
            # ---------- Every frame should have the same initial state --------
            node_pos_curr = BaxterRope.node_pos_curr
            node_radius_curr = BaxterRope.node_radius_curr
            node_quaternion_curr = BaxterRope.node_quaternion_curr

            # ---------- load the information for the center line--------
            if BaxterRope.use_2D_centerline:
                centerline_2d_image = np.load(
                    "../DATA_BaxterRope/centerline_2d_image/%d.npy" %
                    num_frame)
                centerline_2d_image = torch.as_tensor(
                    centerline_2d_image) * 1.0

            centerline_3d_gravity = np.load(
                "../DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame)
            centerline_3d_gravity = torch.as_tensor(
                centerline_3d_gravity).float() * 1.0

            ## ================================================================================
            #     Initial constraints parameters
            ## ================================================================================
            Lambda_strain = torch.zeros((BaxterRope.node_num - 1, 3))
            # Lambda_vol1 = torch.zeros(BaxterRope.node_num)
            # Lambda_radius = torch.zeros(BaxterRope.node_num)
            # Lambda_shape1 = 0

            Lambda_dist = torch.zeros((BaxterRope.node_num - 1, 3))
            Lambda_bending = torch.zeros((BaxterRope.node_num - 2, 4))
            # constraint_q1 = Quaternion([1, 0, 0, 0]).normalised
            # constraint_q2 = Quaternion([1, 0, 0, 0]).normalised

            ## Set actuation point
            # node_pos_curr[BaxterRope.id_node_actuated, :] = BaxterRope.update_pos_actuated
            node_pos_curr[
                BaxterRope.id_node_actuated, :] = centerline_3d_gravity[0]

            ## ================================================================================
            #    Weight Stiffness
            ## ================================================================================
            # wab = 400.0
            # wg = 1.0
            # ws = 100.0
            # wq = 100.0

            wab = 1.0
            wg = 1.0
            ws = 1.0
            wq = 1.0

            # ### ------- Constraints stiffness -------
            # w_gravity = 0.1

            # w_dis = 0.3
            # w_strain = 0.2  ## effect : if on the same plane

            # wq_strain = 1.0
            # wq_bending = 1.0  ## effect : curvature

            # w_SOR = 1.2

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
                # dist_gravity[1:-1, :] = 9.8 * (
                #     BaxterRope.dt**
                #     2) * 0.5 * BaxterRope.gravity_dir * wg * 1 * 1.0

                # -------- calculate the distance move caused by the gravity -------
                dist_gravity = 9.8 * (
                    BaxterRope.dt**
                    2) * 0.5 * BaxterRope.gravity_dir * wg * 1 * 1.0
                #pdb.set_trace()
                # dist_gravity[0] = dist_gravity[0] * 0
                # dist_gravity[-1] = dist_gravity[-1] * 0
                # node_pos_curr = node_pos_curr + (dx_dist * w_dis + dist_gravity * w_gravity) / (w_gravity + w_dis)
                #pdb.set_trace()

                dist_gravity_repeat = dist_gravity.repeat(
                    BaxterRope.node_num, 1)
                dist_gravity_repeat[0] = 0 * dist_gravity_repeat[0]
                dist_gravity_repeat[-1] = 0 * dist_gravity_repeat[-1]
                node_pos_curr = node_pos_curr + (dist_gravity_repeat *
                                                 abs(BaxterRope.w_gravity)) / 1
                # node_pos_curr = node_pos_curr + dist_gravity * w_gravity
                # node_pos_curr = node_pos_curr + dx_dist * w_dis
                # pdb.set_trace()

                for i in range(BaxterRope.LAYERS):

                    ### ------- TDMAL distance constraints -------
                    dx_dist, Lambda_dist, C_dis = Diff_dist.solveDistConstraintTDMALinearSolver(
                        node_pos_curr, Lambda_dist, wab, wab)

                    # node_pos_curr = node_pos_curr + dx_dist * w_dis

                    ### ------- Stretch constraint -------
                    dx_strain, Lambda_strain, dq_strain, C_strain = Diff_stretch.C_strain_solve(
                        node_pos_curr, Lambda_strain, node_quaternion_curr, ws,
                        ws, wq)
                    dx_strain[0] = dx_strain[0] * 0
                    dx_strain[-1] = dx_strain[-1] * 0

                    # node_pos_curr = node_pos_curr + dx_strain * w_strain
                    # node_pos_curr = node_pos_curr + (dx_strain * w_strain + dx_dist * w_dis) / 2

                    ### ------- Bending constraint -------
                    dq_bending, Lambda_bending, C_bending = Diff_bending.solve_BendTwistConstraint(
                        node_pos_curr, node_quaternion_curr, Lambda_bending,
                        wq, wq)

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
                        torch.linalg.norm(node_quaternion_curr, dim=1)).T
                    node_quaternion_curr = torch.nan_to_num(
                        node_quaternion_curr, nan=0)

                    ### radius
                    # node_radius_curr = node_radius_curr + 1.00 * (dr+dr_vol)/2
                    # node_radius_curr = node_radius_curr + (dr_vol1 + dr_vol2 + dr_radius) / 3

            ### ==========================================================================================
            ##         Updating Constraints Solving : Vel/Pos
            ### ==========================================================================================
            # velocity = (node_pos_curr - BaxterRope.node_pos_curr) / BaxterRope.dt
            # qp_conjugate = torch.zeros_like(node_quaternion_curr)
            # qp_conjugate[:, 0] = -BaxterRope.node_quaternion_curr[:, 0]
            # qp_conjugate[:, 1] = -BaxterRope.node_quaternion_curr[:, 1]
            # qp_conjugate[:, 2] = -BaxterRope.node_quaternion_curr[:, 2]
            # qp_conjugate[:, 3] = BaxterRope.node_quaternion_curr[:, 3]
            # PQ, pq0 = BaxterRope.mulQuaternion(qp_conjugate, node_quaternion_curr)

            # BaxterRope.node_pos_curr = node_pos_curr
            # BaxterRope.node_quaternion_curr = node_quaternion_curr
            # # _, _, BaxterRope.node_quaternion_curr = BaxterRope.Quaternion(BaxterRope.node_pos_curr)
            # BaxterRope.node_radius_curr = node_radius_curr

            ### ==========================================================================================
            ##         Get different loss
            ### ==========================================================================================
            #Initial loss

            loss_projection_lineseg_3D = BaxterRope.getProjectionLineSegmentsLoss(
                node_pos_curr.shape[0], centerline_3d_gravity.shape[0],
                node_pos_curr, centerline_3d_gravity)
            print("No. frame:", num_frame)
            print("Loss     :", loss_projection_lineseg_3D)
            node_pos_curr = BaxterRope.node_pos_curr
            node_radius_curr = BaxterRope.node_radius_curr
            node_quaternion_curr = BaxterRope.node_quaternion_curr

            dist_gravity = torch.zeros_like(BaxterRope.node_pos_curr)

            ### ==========================================================================================
            ## define FINAL combination of different losses with weights(The weight for the different frame should be different)
            ### ==========================================================================================
            if num_frame == 1:
                loss_2frames = loss_2frames + loss_projection_lineseg_3D * 1
            else:
                loss_2frames = loss_2frames + loss_projection_lineseg_3D * 1
            # loss = loss_2D_centerline_image_p2p
            # loss = loss_3D_centerline_gravity_p2p

        # BaxterRope.node_pos_curr = node_pos_curr
        # BaxterRope.node_quaternion_curr = node_quaternion_curr
        # # _, _, BaxterRope.node_quaternion_curr = BaxterRope.Quaternion(BaxterRope.node_pos_curr)
        # BaxterRope.node_radius_curr = node_radius_curr

        time_finish_frame = time.time()
        curr_itr_time = time_finish_frame - time_begin_itr
        total_itr_time = time_finish_frame - time_begin_frame
        print("\n")
        print("Curr  itr time :", curr_itr_time)
        print("Total itr time :", total_itr_time)
        print(
            "---------------------------------------------------------------------------------------------"
        )
        print("Total Loss     :", loss_2frames.detach().numpy(), ' itr :', itr)
        # print("Loss 2D centerline image p2p : ", loss_2D_centerline_image_p2p.detach().numpy())
        # print("Loss 2D centerline gravity p2p : ", loss_3D_centerline_gravity_p2p.detach().numpy())
        print(
            "---------------------------------------------------------------------------------------------"
        )
        print("\n")

        ### ==========================================================================================
        ##         Constraint Losses
        ### ==========================================================================================
        # print(C_bending)
        # print(C_dis)
        # print(C_strain)
        # C_dis
        # C_strain

        C_dis_MAX = torch.max(torch.abs(C_dis))
        C_bending_MAX = torch.max(torch.linalg.norm(C_bending, dim=1))
        C_strain_MAX = torch.max(torch.linalg.norm(C_strain, dim=1))

        ### ==========================================================================================
        ### Do loss backpropagation
        ### ==========================================================================================
        # loss.backward(retain_graph=True)
        if BaxterRope.backward == True:
            loss_2frames.backward()
            BaxterRope.optimizer.step()
        #pdb.set_trace()

        # -------- Print Related Updated Parameter --------
        print("w_gravity:", BaxterRope.w_gravity.detach().numpy())
        print("w_dis    :", BaxterRope.w_dis.detach().numpy())
        print("w_strain :", BaxterRope.w_strain.detach().numpy())
        print("wq_strain:", BaxterRope.wq_strain.detach().numpy())
        print("wq_bending:", BaxterRope.wq_bending.detach().numpy())
        print("w_SOR     :", BaxterRope.w_SOR.detach().numpy())

        ### ==========================================================================================
        ### Recording
        ### ==========================================================================================
        save_loss_itr.append(loss_2frames.detach().numpy())
        save_curr_itr_time.append(curr_itr_time)
        save_total_itr_time.append(total_itr_time)
        save_nodes_pos_itr.append(BaxterRope.node_pos_curr.detach().numpy())
        save_nodes_radius_itr.append(
            BaxterRope.node_radius_curr.detach().numpy())

        save_C_dis_MAX_itr.append(C_dis_MAX.detach().numpy())
        save_C_bending_MAX_itr.append(C_bending_MAX.detach().numpy())
        save_C_strain_MAX_itr.append(C_strain_MAX.detach().numpy())

        save_C_dis_itr.append(C_dis.detach().numpy())
        save_C_bending_itr.append(C_bending.detach().numpy())
        save_C_strain_itr.append(C_strain.detach().numpy())

        save_w_gravity_itr.append(BaxterRope.w_gravity.detach().numpy())
        save_w_dis_itr.append(BaxterRope.w_dis.detach().numpy())
        save_w_strain_itr.append(BaxterRope.w_strain.detach().numpy())
        save_wq_strain_itr.append(BaxterRope.wq_strain.detach().numpy())
        save_wq_bending_itr.append(BaxterRope.wq_bending.detach().numpy())
        save_w_SOR_itr.append(BaxterRope.w_SOR.detach().numpy())
    #pdb.set_trace()

    # ### ==========================================================================================
    # ### Plotting
    # ### ==========================================================================================
    # line1_real_3d_data = centerline_3d_gravity.detach().numpy()
    # line1_real_3d.set_data(line1_real_3d_data[:, 0], line1_real_3d_data[:, 1])
    # line1_real_3d.set_3d_properties(line1_real_3d_data[:, 2])

    # line2_sim_3d_data = BaxterRope.node_pos_curr.detach().numpy()
    # line2_sim_3d.set_data(line2_sim_3d_data[:, 0], line2_sim_3d_data[:, 1])
    # line2_sim_3d.set_3d_properties(line2_sim_3d_data[:, 2])

    # line3_loss.set_data(np.arange(len(plot_loss_his)), plot_loss_his)
    # # plt.show()
    # plt.pause(0.01)
    # # pdb.set_trace()

    # np.save(
    #     BaxterRope.save_dir + "frame_{}_itr_{}.npy".format(num_frame, itr),
    #     np.hstack(
    #         (BaxterRope.node_pos_curr.detach().numpy(), BaxterRope.node_radius_curr.reshape(-1,
    #                                                                                         1).detach().numpy(),
    #          np.zeros((BaxterRope.node_pos_curr.shape[0], 1)) + curr_itr_time,
    #          np.zeros((BaxterRope.node_pos_curr.shape[0], 1)) + total_itr_time)))

    ### ==========================================================================================
    ### Plotting
    # ### ==========================================================================================
    line1_real_3d_data = centerline_3d_gravity.detach().numpy()
    line1_real_3d.set_data(line1_real_3d_data[:, 0], line1_real_3d_data[:, 1])
    line1_real_3d.set_3d_properties(line1_real_3d_data[:, 2])

    # scatter1_sim_3d._offsets3d = (line1_real_3d_data[:, 0], line1_real_3d_data[:, 1], line1_real_3d_data[:, 2])

    line2_sim_3d_data = BaxterRope.node_pos_curr.detach().numpy()
    line2_sim_3d.set_data(line2_sim_3d_data[:, 0], line2_sim_3d_data[:, 1])
    line2_sim_3d.set_3d_properties(line2_sim_3d_data[:, 2])

    scatter1_sim_3d._offsets3d = (line2_sim_3d_data[:, 0],
                                  line2_sim_3d_data[:,
                                                    1], line2_sim_3d_data[:,
                                                                          2])

    line3_loss.set_data(np.arange(len(save_loss_itr)), save_loss_itr)

    line4_itr_C_dis_max.set_data(np.arange(len(save_C_dis_MAX_itr)),
                                 save_C_dis_MAX_itr)
    line5_itr_C_bending_max.set_data(np.arange(len(save_C_bending_MAX_itr)),
                                     save_C_bending_MAX_itr)
    line6_itr_C_strain_max.set_data(np.arange(len(save_C_strain_MAX_itr)),
                                    save_C_strain_MAX_itr)

    plt.show()
    # plt.pause(1.0)
    # pdb.set_trace()

    # ### ==========================================================================================
    # ### SAVE DATA
    save_loss_itr = np.array(save_loss_itr)
    save_curr_itr_time = np.array(save_curr_itr_time)
    save_total_itr_time = np.array(save_total_itr_time)
    save_nodes_pos_itr = np.array(save_nodes_pos_itr)
    save_nodes_radius_itr = np.array(save_nodes_radius_itr)

    save_C_dis_itr = np.array(save_C_dis_itr)
    save_C_bending_itr = np.array(save_C_bending_itr)
    save_C_strain_itr = np.array(save_C_strain_itr)

    save_w_gravity_itr = np.array(save_w_gravity_itr)
    save_w_dis_itr = np.array(save_w_dis_itr)
    save_w_strain_itr = np.array(save_w_strain_itr)
    save_wq_strain_itr = np.array(save_wq_strain_itr)
    save_wq_bending_itr = np.array(save_wq_bending_itr)
    save_w_SOR_itr = np.array(save_w_SOR_itr)

    min_total_itr_loss_id = np.argmin(
        save_loss_itr,
        axis=0)  ## only first min id is returned by using np.argmin
    # #pdb.set_trace()
    weight_result = np.zeros((BaxterRope.node_num, 1))
    weight_result[:6] = np.array([
        save_w_gravity_itr[min_total_itr_loss_id],
        save_w_dis_itr[min_total_itr_loss_id],
        save_w_strain_itr[min_total_itr_loss_id],
        save_wq_strain_itr[min_total_itr_loss_id],
        save_wq_bending_itr[min_total_itr_loss_id],
        save_w_SOR_itr[min_total_itr_loss_id]
    ]).reshape(6, 1)
    save_optimized_data = np.hstack((
        weight_result,
        save_nodes_pos_itr[min_total_itr_loss_id],
        save_nodes_radius_itr[min_total_itr_loss_id].reshape(-1, 1),
        np.zeros((BaxterRope.node_num, 1)) +
        save_curr_itr_time[min_total_itr_loss_id],
        np.zeros((BaxterRope.node_num, 1)) +
        save_total_itr_time[min_total_itr_loss_id],
        np.zeros((BaxterRope.node_num, 1)) + min_total_itr_loss_id,
        np.zeros(
            (BaxterRope.node_num, 1)) + save_loss_itr[min_total_itr_loss_id],
        ## the following 3 are max constraints convergency over each iteration
        np.zeros((BaxterRope.node_num, 1)) +
        np.insert(save_C_dis_itr[min_total_itr_loss_id], 19, 0),
        np.zeros((BaxterRope.node_num, 1)) +
        np.insert(save_C_dis_itr[min_total_itr_loss_id], 19, [0, 0]),
        np.zeros((BaxterRope.node_num, 1)) +
        np.insert(save_C_strain_itr[min_total_itr_loss_id], 19, 0)))

    np.save(BaxterRope.save_dir + "para_estimation.npy", save_optimized_data)

    pdb.set_trace()