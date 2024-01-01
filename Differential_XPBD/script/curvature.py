import numpy as np
import pdb
import torch
import matplotlib.pyplot as plt

# refer https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy


# -------- Get the lowest point index along the gravity --------
def ProjectionPointAlongGravity(centerline_3d_gravity, node_pos_curr,
                                gravity_dir):

    extend_gravity_scale = 1
    gravity = torch.vstack(
        (centerline_3d_gravity[0] - gravity_dir * extend_gravity_scale,
         centerline_3d_gravity[0] + gravity_dir * extend_gravity_scale))
    gravity_vector = gravity[1] - gravity[0]
    gravity_length_square = torch.linalg.norm(gravity_vector)**2

    sim_proj_gravity = torch.sum((node_pos_curr - gravity[0]) * gravity_vector,
                                 dim=1)
    sim_proj_gravity = sim_proj_gravity / gravity_length_square
    sim_lowest_gravity_id = torch.max(sim_proj_gravity, dim=0)

    return sim_lowest_gravity_id.indices


#-------- Get the curvature from the simulation node --------
def GetCurvatureForSimnode(sim_node):
    # refer https://stackoverflow.com/questions/28269379/curve-curvature-in-numpy

    # -------- Calculate the first order gradient for the simulation nodes --------
    dx_dt = torch.as_tensor(torch.gradient(sim_node[:, 0])[0])
    dy_dt = torch.as_tensor(torch.gradient(sim_node[:, 1])[0])
    dz_dt = torch.as_tensor(torch.gradient(sim_node[:, 2])[0])
    #pdb.set_trace()

    #  -------- Calculate the second order gradient for the simulation nodes --------
    #d2s_dt2 = torch.as_tensor(torch.gradient(ds_dt)[0])
    d2x_dt2 = torch.as_tensor(torch.gradient(dx_dt)[0])
    d2y_dt2 = torch.as_tensor(torch.gradient(dy_dt)[0])
    d2z_dt2 = torch.as_tensor(torch.gradient(dz_dt)[0])

    #ref https://blog.csdn.net/weixin_42040046/article/details/97760083
    #sqrt((x2^2+y2^2+z2^2)*(x1^2+y1^2+z1^2)-(x1*x2+y1*y2+z1*z2)^2)/((sqrt(x1^2+y1^2+z1^2))^3)
    curvature = torch.sqrt(
        (d2x_dt2**2 + d2y_dt2**2 + d2z_dt2**2) *
        (dx_dt**2 + dy_dt**2 + dz_dt**2) -
        (d2x_dt2 * dx_dt + d2y_dt2 * dy_dt + d2z_dt2 * dz_dt)**2) / (
            dx_dt * dx_dt + dy_dt * dy_dt + dz_dt * dz_dt)**1.5

    return curvature


if __name__ == '__main__':
    curvature_list = []
    curvature_list_index = []
    for num_frame in range(16, 66):
        # ------- Load simulation node -------
        sim_node = torch.as_tensor(
            np.load(
                "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/frame_%d.npy"
                % num_frame))[:, :3]
        centerline_3d_gravity = torch.as_tensor(
            np.load(
                "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy" %
                num_frame))

        gravity_dir = torch.as_tensor([0.10381715, 0.9815079, -0.16082364])

        # ------- The index of lowest point along the gravity -------
        lowest_index = ProjectionPointAlongGravity(centerline_3d_gravity,
                                                   sim_node, gravity_dir)

        # ------- Get the curvature for the simulation nodes -------
        curvature = GetCurvatureForSimnode(sim_node)
        curvature = torch.nan_to_num(curvature, nan=0.0)
        curvature_list.append(torch.max(curvature).float(
        ))  # The maximum curvature as the curvature of the line
        #curvature_list.append(curvature[lowest_index].float())
        #curvature_list.append(torch.mean(curvature).float())
        curvature_list_index.append(torch.argmax(curvature))
    np.save("../DATA/curvature.npy", curvature_list)
    plt.plot(torch.arange(16, 66, 1), curvature_list)
    plt.show()
