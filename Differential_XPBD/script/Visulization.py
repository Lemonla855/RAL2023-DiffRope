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

import sys
from matplotlib import cm

import sys

from numpy.core.defchararray import array

sys.path.append("../../../")

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm

import pdb
#sys.path.append("../../../")
sys.path.append("../../")
from PyElastica_PBDRope.examples.Visualization._povmacros import Stages, pyelastica_rod, render

# from PyElastica_PBDRope.examples.Visualization.Rope_Rendering import pbd_rope_render

import sys

from numpy.core.defchararray import array

sys.path.append("..")

import multiprocessing
import os
from functools import partial
from multiprocessing import Pool

import numpy as np
from moviepy.editor import ImageSequenceClip
from scipy import interpolate
from tqdm import tqdm

import pdb

#from PyElastica_PBDRope.examples.Visualization._povmacros import Stages, pyelastica_rod, render


class VisulizationTool():
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

    def WriteVideo(self, frame_start, frame_end, fps, writer_path,
                   image_file_path):
        writer = imageio.get_writer(writer_path, fps=5)

        for index in range(frame_start, frame_end):

            image_path = image_file_path + "/%d.png" % index
            frame = imageio.imread(image_path)
            writer.append_data(frame)

        writer.close()


## ================================================================================
#    Draw Plot for raw RGB image and 2D simulation result
## ================================================================================
class RGBPBD2D():
    def RealtoSimFigure(self, loss_type, RealtoSimFigure_name):

        # # --------Initialize the Figure --------.
        # fig_curve = plt.figure()
        # ax_curve = Axes3D(fig_curve)

        # # -------- Add label to the axes --------
        # ax_curve.set_xlabel('x')
        # ax_curve.set_ylabel('y')
        # ax_curve.set_zlabel('z')

        # -------- Initialize the Visulization Tool  --------
        vistool = VisulizationTool()

        # -------- Initialize the File  --------
        #loss_type = "projection"  ##The only place  need to modify
        path = '../DATA/save_loss_withConstConvergency/' + loss_type
        #Read all the file in the path
        #pdb.set_trace()
        main_files = os.listdir(path)
        #pdb.set_trace()

        #Create file to save all the plotting
        plotting_file_path = "../DATA/RealtoSimPlotting/" + loss_type + "/plot_2D"
        if not os.path.exists(plotting_file_path):
            os.makedirs(plotting_file_path)

        #The path of original RBG Image
        rgb_path = "../DATA/DATA_BaxterRope/rgb_raw_Baxter"
        centerline_3D_gravity_path = "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean"
        #pdb.set_trace()
        #-------- Information for video writer --------
        frame_start = 16
        # frame_end = 66
        fps = 5
        subfile_plotting_file = "../DATA/RealtoSimPlotting/" + loss_type
        if not os.path.exists(subfile_plotting_file):
            os.makedirs(subfile_plotting_file)

        for file_LossName in main_files:
            #pdb.set_trace()

            loss_file_path = path + "/" + file_LossName

            if os.path.splitext(file_LossName)[1] == ".npy":

                frame_index = os.path.splitext(file_LossName)[0].split('_')[
                    1]  #The index of the frame from npy file

                #-------- Read the data from npy file --------
                sim_node = torch.as_tensor(
                    np.load(loss_file_path)[:, :3]).float()

                #-------- Plot the Simulation Node result by projecting onto the 2D image--------
                plt.axis("off")
                plt.title(str(RealtoSimFigure_name))  # The name of the figure
                sim_node_2D = vistool.ProjectionPtfrom3DNode(sim_node)
                plt.scatter(sim_node_2D[:, 0], sim_node_2D[:, 1])
                plt.plot(sim_node_2D[:, 0], sim_node_2D[:, 1])

                # -------- load 3D centerline and plot the centerline by projecting onto the 2S--------
                centerline_3D_gravity = torch.as_tensor(
                    np.load(centerline_3D_gravity_path + "/" + frame_index +
                            ".npy")).float()
                centerline_2D = vistool.ProjectionPtfrom3DNode(
                    centerline_3D_gravity
                )  # This kind of projection can be applied to 3D centerline or 3D sim node
                #Plot the 3D centerline
                #pdb.set_trace()

                # -------- load 2D centerline which projects from 3D centerline --------
                plt.scatter(centerline_2D[:, 0], centerline_2D[:, 1], s=0.01)
                plt.plot(centerline_2D[:, 0], centerline_2D[:, 1])

                #-------- Plot corresponding rbg image --------
                RgbImage_path = rgb_path + "/" + frame_index + ".png"
                RgbImage = plt.imread(RgbImage_path)
                plt.imshow(RgbImage)

                #--------- The path to save the plot ---------#
                #pdb.set_trace()
                plt.savefig(subfile_plotting_file + "/" + frame_index + ".png")
                plt.cla()

        # ------- write the video -------
        writer_path = "../DATA/video/video_2d/" + loss_type
        if not os.path.exists(writer_path):
            os.makedirs(writer_path)
        writer_location = "../DATA/video/video_2d/" + loss_type + "/" + RealtoSimFigure_name + ".mp4"
        image_file_path = subfile_plotting_file
        frame_end = frame_start + len(
            main_files) - 1  #The endFrame is different
        vistool.WriteVideo(frame_start, frame_end, fps, writer_location,
                           image_file_path)
        #pdb.set_trace()


class PyElasticRendering():
    def ExecuteRendering(self, DATA_PATH, OUTPUT_VIDEO_Path, VIDEO_NAME,
                         OUTPUT_IMAGES_DIR):
        #-------- Setup (USER DEFINE) --------
        DATA_PATH = DATA_PATH  # Path to the simulation data
        SAVE_PICKLE = True

        # -------- Rendering Configuration (USER DEFINE) --------
        OUTPUT_VIDEO_Path = OUTPUT_VIDEO_Path
        OUTPUT_IMAGES_DIR = OUTPUT_IMAGES_DIR
        #OUTPUT_FILENAME = "pov_heartshape"
        #OUTPUT_IMAGES_DIR = "frames_heartshape"
        FPS = 20.0
        WIDTH = 1920  # 400
        HEIGHT = 1080  # 250
        DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']
        TOTAL_FRAMES = 35  ##The numbder of frame
        START_FRAME = 30

        # -------- If the file not exits --------
        if not os.path.exists(OUTPUT_IMAGES_DIR):
            os.makedirs(OUTPUT_IMAGES_DIR)
        if not os.path.exists(OUTPUT_VIDEO_Path):
            os.makedirs(OUTPUT_VIDEO_Path)
        # output video path and name
        OUTPUT_VIDEO_PATH_NAME = OUTPUT_VIDEO_Path + "/" + VIDEO_NAME

        #pdb.set_trace()

        # Camera/Light Configuration (USER DEFINE)
        stages = Stages()
        # stages.add_camera(
        #     # Add diagonal viewpoint
        #     location=[40.0, 100.5, -40.0],
        #     angle=30,
        #     look_at=[4.0, 2.7, 2.0],
        #     name="diag",
        # )
        # stages.add_camera(
        #     # Add top viewpoint
        #     location=[0, 200, 3],
        #     angle=30,
        #     look_at=[0.0, 0, 3],
        #     sky=[-1, 0, 0],
        #     name="top",
        # )

        stages.add_camera(
            # Add diagonal viewpoint
            location=[0.0, 20., -30.0],
            angle=50,
            look_at=[4.0, 2.7, 2.0],
            name="diag",
        )
        stages.add_camera(
            # Add top viewpoint
            location=[0, 200, 3],
            angle=30,
            look_at=[0.0, 0, 0],
            sky=[-1, 0, 0],
            name="top",
        )
        stages.add_light(
            # Sun light
            position=[1500, 2500, -1000],
            color="White",
            camera_id=-1,
        )
        stages.add_light(
            # Flash light for camera 0
            position=[15.0, 10.5, -15.0],
            color=[0.09, 0.09, 0.1],
            camera_id=0,
        )
        stages.add_light(
            # Flash light for camera 1
            position=[0.0, 8.0, 5.0],
            color=[0.09, 0.09, 0.1],
            camera_id=1,
        )
        stage_scripts = stages.generate_scripts()

        # Externally Including Files (USER DEFINE)
        # If user wants to include other POVray objects such as grid or coordinate axes,
        # objects can be defined externally and included separately.
        included = ["../PyElastica_PBDRope/examples/Visualization/default.inc"]

        # Multiprocessing Configuration (USER DEFINE)
        MULTIPROCESSING = True
        THREAD_PER_AGENT = 4  # Number of thread use per rendering process.
        NUM_AGENT = multiprocessing.cpu_count(
        ) // 2  # number of parallel rendering.

        assert os.path.exists(DATA_PATH), "File does not exists"

        dir = np.array([0.38047181, 0.09438525, 0.8950117])

        for i in range(START_FRAME, START_FRAME + TOTAL_FRAMES):

            centerline_3d_gravity_clean = np.ones((42, 3)) * dir[None, :]
            centerline_3d_gravity_clean = centerline_3d_gravity_clean.T
            file_name = DATA_PATH + 'frame_' + str(i + 1) + '.npy'
            try:
                with open(file_name, "rb") as fptr:
                    data = np.load(fptr)
                    # pdb.set_trace()

                    data = np.moveaxis(
                        data, -1, 0
                    )  # shape: (2, 4, num_element) --- only shape[1, :, :] is useful

                    # rot_mtx = np.array([[1.0000000, 0.0000000, 0.0000000],
                    #                     [0.0000000, 0.0000000, -1.0000000],
                    #                     [0.0000000, 1.0000000, 0.0000000]])
                    rot_mtx = np.array([[1.0000000, 0.0000000, 0.0000000],
                                        [0.0000000, 1.0000000, 0.0000000],
                                        [0.0000000, 0.0000000, 1.0000000]])

                    data_rot = np.matmul(rot_mtx, data[0:3, :])
                    data_rot[1, :] = -data_rot[1, :]
                    data_rot = data_rot * 30
                    # data_rot[0,:]= data_rot[0,:]-np.min(data_rot[0,:])
                    # data_rot[1,:]=data_rot[2,:]-np.min(data_rot[2,:])
                    # data_rot[2,:]=data_rot[1,:]-np.min(data_rot[1,:])
                    # data_rot[0,:]= data_rot[0,:]-np.min(data_rot[0,:])
                    # data_rot[2,:]=data_rot[2,:]-np.min(data_rot[2,:])
                    # data_rot[1,:]=data_rot[1,:]-np.min(data_rot[1,:])

                    data_rot[0, :] = data_rot[0, :] - (-3)
                    data_rot[2, :] = data_rot[2, :] - 25
                    data_rot[1, :] = data_rot[1, :] - (-10) + 2
                    #print(np.min(data_rot[0,:]),np.min(data_rot[2,:]),np.min(data_rot[1,:]))

                    data_radius = data[3, :] * 20  # (TODO) radius could change

            except OSError as err:
                print("Cannot open the datafile {}".format(DATA_PATH))
                print(str(err))
                raise

            # -------- Add 3D centerline -------
            file_name = "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean_downsample/%d.npy" % (
                i + 1)
            centerline_3d_gravity = np.load(file_name)
            centerline_3d_gravity = np.moveaxis(centerline_3d_gravity, -1, 0)

            centerline_3d_gravity_clean[:, :centerline_3d_gravity.
                                        shape[1]] = centerline_3d_gravity

            centerline_3d_gravity_clean = np.matmul(
                rot_mtx, centerline_3d_gravity_clean[0:3, :])
            centerline_3d_gravity_clean[
                1, :] = -centerline_3d_gravity_clean[1, :]
            centerline_3d_gravity_clean = centerline_3d_gravity_clean * 30

            centerline_3d_gravity_clean[
                0, :] = centerline_3d_gravity_clean[0, :] - (-3)
            centerline_3d_gravity_clean[
                2, :] = centerline_3d_gravity_clean[2, :] - 25
            centerline_3d_gravity_clean[
                1, :] = centerline_3d_gravity_clean[1, :] - (-10) + 2

            centerline_3d_gravity_clean_radius = np.ones(
                centerline_3d_gravity_clean.shape[1]) * data[3, 0] * 12

            if i == 30:
                times = np.array([0.0])  # shape: (timelength)
                xs = np.expand_dims(data_rot, axis=0)
                base_radius = data_radius

                centerline_3d_xs = np.expand_dims(centerline_3d_gravity_clean,
                                                  axis=0)
                centerline_3d_radius = centerline_3d_gravity_clean_radius

            else:
                times = np.hstack((times, np.array([i])))
                xs = np.vstack((xs, np.expand_dims(data_rot, axis=0)))
                base_radius = np.vstack((base_radius, data_radius))

                centerline_3d_xs = np.vstack(
                    (centerline_3d_xs,
                     np.expand_dims(centerline_3d_gravity_clean, axis=0)))
                centerline_3d_radius = np.vstack(
                    (centerline_3d_radius, centerline_3d_gravity_clean_radius))

        # pdb.set_trace()

        # Convert data to numpy array
        # times = np.array(data["time"])  # shape: (timelength)
        # xs = np.array(data["position"])  # shape: (timelength, 3, num_element)
        # times = np.array([1.0])  # shape: (timelength)
        # xs = np.expand_dims(data[1, :, 0:3], axis=0)

        # Interpolate Data
        # Interpolation step serves two purposes. If simulated frame rate is lower than
        # the video frame rate, the intermediate frames are linearly interpolated to
        # produce smooth video. Otherwise if simulated frame rate is higher than
        # the video frame rate, interpolation reduces the number of frame to reduce
        # the rendering time.
        runtime = times.max()  # Physical run time
        # total_frame = int(runtime * FPS)  # Number of frames for the video
        recorded_frame = times.shape[0]  # Number of simulated frames
        total_frame = recorded_frame  # Number of frames for the video
        times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale
        # pdb.set_trace()

        # xs = interpolate.interp1d(times, xs, axis=0)(times_true)
        # times = interpolate.interp1d(times, times, axis=0)(times_true)

        # base_radius = np.ones_like(xs[:, 0, :]) * 0.50  # (TODO) radius could change
        # pdb.set_trace()

        # Rendering
        # For each frame, a 'pov' script file is generated in OUTPUT_IMAGE_DIR directory.
        batch = []
        for view_name in stage_scripts.keys():  # Make Directory
            output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
            os.makedirs(output_path, exist_ok=True)
        for frame_number in tqdm(range(total_frame), desc="Scripting"):
            for view_name, stage_script in stage_scripts.items():
                output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)

                # Colect povray scripts
                script = []
                script.extend(['#include "{}"'.format(s) for s in included])
                script.append(stage_script)

                # If the data contains multiple rod, this part can be modified to include
                # multiple rods.
                rod_object = pyelastica_rod(
                    x=xs[frame_number],
                    r=base_radius[frame_number],
                    #transmit=0.3,
                    color="rgb<224/255,36/255,1/255>",
                    #color="rgb<0.45,0.39,1>",
                )
                script.append(rod_object)

                rod_object = pyelastica_rod(
                    x=centerline_3d_xs[frame_number],
                    r=centerline_3d_radius[frame_number],
                    color="rgb<0.45,0.39,1>",
                    #color="rgb<1.0,0.0,0.0>",
                )
                #pdb.set_trace()
                script.append(rod_object)
                pov_script = "\n".join(script)

                # Write .pov script file
                file_path = os.path.join(output_path,
                                         "frame_{:04d}".format(frame_number))
                with open(file_path + ".pov", "w+") as f:
                    f.write(pov_script)
                batch.append(file_path)

        # Process POVray
        # For each frames, a 'png' image file is generated in OUTPUT_IMAGE_DIR directory.
        #pdb.set_trace()
        pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
        if MULTIPROCESSING:
            func = partial(
                render,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=THREAD_PER_AGENT,
            )
            with Pool(NUM_AGENT) as p:
                for message in p.imap_unordered(func, batch):
                    # (TODO) POVray error within child process could be an issue
                    pbar.update()
        else:
            for filename in batch:
                render(
                    filename,
                    width=WIDTH,
                    height=HEIGHT,
                    display=DISPLAY_FRAMES,
                    pov_thread=multiprocessing.cpu_count(),
                )
                pbar.update()

        # Create Video using moviepy
        for view_name in stage_scripts.keys():
            imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
            imageset = [
                os.path.join(imageset_path, path)
                for path in os.listdir(imageset_path) if path[-3:] == "png"
            ]
            imageset.sort()
            filename = OUTPUT_VIDEO_PATH_NAME + "_" + view_name + ".mp4"

            FPS = 3
            #pdb.set_trace()
            clip = ImageSequenceClip(imageset, fps=FPS)
            clip.write_videofile(filename, fps=10)


## ================================================================================
#    PyElastic for multi rods
## ================================================================================
class PyElasticRenderingMultiRod():
    def ExecuteRendering(self, DATA_PATH, OUTPUT_VIDEO_Path, VIDEO_NAME,
                         OUTPUT_IMAGES_DIR):
        #-------- Setup (USER DEFINE) --------
        DATA_PATH = DATA_PATH  # Path to the simulation data
        SAVE_PICKLE = True

        # -------- Rendering Configuration (USER DEFINE) --------
        OUTPUT_VIDEO_Path = OUTPUT_VIDEO_Path
        OUTPUT_IMAGES_DIR = OUTPUT_IMAGES_DIR
        #OUTPUT_FILENAME = "pov_heartshape"
        #OUTPUT_IMAGES_DIR = "frames_heartshape"
        FPS = 20.0
        WIDTH = 1920  # 400
        HEIGHT = 1080  # 250
        DISPLAY_FRAMES = "Off"  # Display povray images during the rendering. ['On', 'Off']
        TOTAL_FRAMES = 35  ##The numbder of frame
        START_FRAME = 30

        # -------- If the file not exits --------
        if not os.path.exists(OUTPUT_IMAGES_DIR):
            os.makedirs(OUTPUT_IMAGES_DIR)
        if not os.path.exists(OUTPUT_VIDEO_Path):
            os.makedirs(OUTPUT_VIDEO_Path)
        # output video path and name
        OUTPUT_VIDEO_PATH_NAME = OUTPUT_VIDEO_Path + "/" + VIDEO_NAME

        #pdb.set_trace()

        # Camera/Light Configuration (USER DEFINE)
        stages = Stages()
        # stages.add_camera(
        #     # Add diagonal viewpoint
        #     location=[40.0, 100.5, -40.0],
        #     angle=30,
        #     look_at=[4.0, 2.7, 2.0],
        #     name="diag",
        # )
        # stages.add_camera(
        #     # Add top viewpoint
        #     location=[0, 200, 3],
        #     angle=30,
        #     look_at=[0.0, 0, 3],
        #     sky=[-1, 0, 0],
        #     name="top",
        # )

        stages.add_camera(
            # Add diagonal viewpoint
            location=[0.0, 25., -150.0],  # 54,
            #location=[0.0, 25., -100.0] ,# 54,
            angle=30,
            look_at=[4.0, 2.7, 2.0],
            name="diag",
        )
        stages.add_camera(
            # Add top viewpoint
            location=[0, 200, 3],
            angle=30,
            look_at=[0.0, 0, 0],
            sky=[-1, 0, 0],
            name="top",
        )
        stages.add_light(
            # Sun light
            position=[1500, 2500, -1000],
            color="White",
            camera_id=-1,
        )
        stages.add_light(
            # Flash light for camera 0
            position=[15.0, 10.5, -15.0],
            color=[0.09, 0.09, 0.1],
            camera_id=0,
        )
        stages.add_light(
            # Flash light for camera 1
            position=[0.0, 8.0, 5.0],
            color=[0.09, 0.09, 0.1],
            camera_id=1,
        )
        stage_scripts = stages.generate_scripts()

        # Externally Including Files (USER DEFINE)
        # If user wants to include other POVray objects such as grid or coordinate axes,
        # objects can be defined externally and included separately.
        included = ["../PyElastica_PBDRope/examples/Visualization/default.inc"]

        # Multiprocessing Configuration (USER DEFINE)
        MULTIPROCESSING = True
        THREAD_PER_AGENT = 1  # Number of thread use per rendering process.
        NUM_AGENT = multiprocessing.cpu_count(
        ) // 2  # number of parallel rendering.

        #assert os.path.exists(DATA_PATH), "File does not exists"

        dir = np.array([0.38047181, 0.09438525, 0.8950117])
        rendering_frame = np.array([30, 53])
        #DATA_PATHS = np.array(["../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/","../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/"])
        DATA_PATHS = np.array([
            "../DATA/save_loss_withConstConvergency/XPBD/projection/centerline_3d_gravity_curve/"
        ])
        DATA_PATH = "../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/"

        #DATA_PATHS = np.array(["../DATA/save_loss_withConstConvergency/projection/centerline_3d_gravity/"])
        #for index,DATA_PATH in enumerate (DATA_PATHS):
        #i=40
        for index, i in enumerate(rendering_frame):

            centerline_3d_gravity_clean = np.ones((42, 3)) * dir[None, :]
            centerline_3d_gravity_clean = centerline_3d_gravity_clean.T
            file_name = DATA_PATH + 'frame_' + str(i + 1) + '.npy'
            try:
                with open(file_name, "rb") as fptr:
                    data = np.load(fptr)
                    # pdb.set_trace()

                    data = np.moveaxis(
                        data, -1, 0
                    )  # shape: (2, 4, num_element) --- only shape[1, :, :] is useful

                    # rot_mtx = np.array([[1.0000000, 0.0000000, 0.0000000],
                    #                     [0.0000000, 0.0000000, -1.0000000],
                    #                     [0.0000000, 1.0000000, 0.0000000]])
                    rot_mtx = np.array([[1.0000000, 0.0000000, 0.0000000],
                                        [0.0000000, 1.0000000, 0.0000000],
                                        [0.0000000, 0.0000000, 1.0000000]])

                    data_rot = np.matmul(rot_mtx, data[0:3, :])
                    data_rot[1, :] = -data_rot[1, :]
                    data_rot = data_rot * 30
                    # data_rot[0,:]= data_rot[0,:]-np.min(data_rot[0,:])
                    # data_rot[1,:]=data_rot[2,:]-np.min(data_rot[2,:])
                    # data_rot[2,:]=data_rot[1,:]-np.min(data_rot[1,:])
                    # data_rot[0,:]= data_rot[0,:]-np.min(data_rot[0,:])
                    # data_rot[2,:]=data_rot[2,:]-np.min(data_rot[2,:])
                    # data_rot[1,:]=data_rot[1,:]-np.min(data_rot[1,:])

                    data_rot[0, :] = data_rot[0, :] - (-3)
                    data_rot[2, :] = data_rot[2, :] - 25
                    data_rot[1, :] = data_rot[1, :] - (-10) + 2
                    #print(np.min(data_rot[0,:]),np.min(data_rot[2,:]),np.min(data_rot[1,:]))

                    data_radius = data[3, :] * 20  # (TODO) radius could change

            except OSError as err:
                print("Cannot open the datafile {}".format(DATA_PATH))
                print(str(err))
                raise

            # -------- Add 3D centerline -------
            file_name = "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean_downsample/%d.npy" % (
                i + 1)
            centerline_3d_gravity = np.load(file_name)
            centerline_3d_gravity = np.moveaxis(centerline_3d_gravity, -1, 0)

            centerline_3d_gravity_clean[:, :centerline_3d_gravity.
                                        shape[1]] = centerline_3d_gravity

            centerline_3d_gravity_clean = np.matmul(
                rot_mtx, centerline_3d_gravity_clean[0:3, :])
            centerline_3d_gravity_clean[
                1, :] = -centerline_3d_gravity_clean[1, :]
            centerline_3d_gravity_clean = centerline_3d_gravity_clean * 30

            centerline_3d_gravity_clean[
                0, :] = centerline_3d_gravity_clean[0, :] - (-3)
            centerline_3d_gravity_clean[
                2, :] = centerline_3d_gravity_clean[2, :] - 25
            centerline_3d_gravity_clean[
                1, :] = centerline_3d_gravity_clean[1, :] - (-10) + 2

            centerline_3d_gravity_clean_radius = np.ones(
                centerline_3d_gravity_clean.shape[1]) * data[3, 0] * 6

            if index == 0:
                times = np.array([0.0])  # shape: (timelength)
                xs = np.expand_dims(data_rot, axis=0)
                base_radius = data_radius

                centerline_3d_xs = np.expand_dims(centerline_3d_gravity_clean,
                                                  axis=0)
                centerline_3d_radius = centerline_3d_gravity_clean_radius

            else:
                times = np.hstack((times, np.array([i])))
                xs = np.vstack((xs, np.expand_dims(data_rot, axis=0)))
                base_radius = np.vstack((base_radius, data_radius))

                centerline_3d_xs = np.vstack(
                    (centerline_3d_xs,
                     np.expand_dims(centerline_3d_gravity_clean, axis=0)))
                centerline_3d_radius = np.vstack(
                    (centerline_3d_radius, centerline_3d_gravity_clean_radius))

        # pdb.set_trace()

        # Convert data to numpy array
        # times = np.array(data["time"])  # shape: (timelength)
        # xs = np.array(data["position"])  # shape: (timelength, 3, num_element)
        # times = np.array([1.0])  # shape: (timelength)
        # xs = np.expand_dims(data[1, :, 0:3], axis=0)

        # Interpolate Data
        # Interpolation step serves two purposes. If simulated frame rate is lower than
        # the video frame rate, the intermediate frames are linearly interpolated to
        # produce smooth video. Otherwise if simulated frame rate is higher than
        # the video frame rate, interpolation reduces the number of frame to reduce
        # the rendering time.
        runtime = times.max()  # Physical run time
        # total_frame = int(runtime * FPS)  # Number of frames for the video
        recorded_frame = times.shape[0]  # Number of simulated frames
        total_frame = 1  # Number of frames for the video
        times_true = np.linspace(0, runtime, total_frame)  # Adjusted timescale
        # pdb.set_trace()

        # xs = interpolate.interp1d(times, xs, axis=0)(times_true)
        # times = interpolate.interp1d(times, times, axis=0)(times_true)

        # base_radius = np.ones_like(xs[:, 0, :]) * 0.50  # (TODO) radius could change
        # pdb.set_trace()

        # Rendering
        # For each frame, a 'pov' script file is generated in OUTPUT_IMAGE_DIR directory.
        batch = []
        for view_name in stage_scripts.keys():  # Make Directory
            output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
            os.makedirs(output_path, exist_ok=True)
        for frame_number in tqdm(range(total_frame), desc="Scripting"):
            for view_name, stage_script in stage_scripts.items():
                output_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)

                # Colect povray scripts
                script = []
                script.extend(['#include "{}"'.format(s) for s in included])
                script.append(stage_script)

                # If the data contains multiple rod, this part can be modified to include
                # multiple rods.
                #pdb.set_trace()
                for i in range(xs.shape[0]):
                    # if i == 1:
                    #     color="rgb<0.45,0.39,1>"
                    # else:
                    #     color="rgb<0.45,0.39,11.0,0.0,0.0>"
                    rod_object = pyelastica_rod(
                        x=xs[i],
                        r=base_radius[i],
                        transmit=0.5 * (1 - i),
                        color="rgb<1.0,0.7215686274509804,0.18823529411764706>",
                        #color="rgb<0.45,0.39,1>",
                        #color
                    )
                    script.append(rod_object)

                    rod_object = pyelastica_rod(
                        x=centerline_3d_xs[i],
                        r=centerline_3d_radius[i],
                        color="rgb<0.45,0.39,1>",
                        #color="rgb<1.0,0.0,0.0>",
                    )
                    #pdb.set_trace()
                    script.append(rod_object)
                pov_script = "\n".join(script)

                # Write .pov script file
                file_path = os.path.join(output_path,
                                         "frame_{:04d}".format(frame_number))
                with open(file_path + ".pov", "w+") as f:
                    f.write(pov_script)
                batch.append(file_path)

        # Process POVray
        # For each frames, a 'png' image file is generated in OUTPUT_IMAGE_DIR directory.
        #pdb.set_trace()
        pbar = tqdm(total=len(batch), desc="Rendering")  # Progress Bar
        if MULTIPROCESSING:
            func = partial(
                render,
                width=WIDTH,
                height=HEIGHT,
                display=DISPLAY_FRAMES,
                pov_thread=THREAD_PER_AGENT,
            )
            with Pool(NUM_AGENT) as p:
                for message in p.imap_unordered(func, batch):
                    # (TODO) POVray error within child process could be an issue
                    pbar.update()
        else:
            for filename in batch:
                render(
                    filename,
                    width=WIDTH,
                    height=HEIGHT,
                    display=DISPLAY_FRAMES,
                    pov_thread=multiprocessing.cpu_count(),
                )
                pbar.update()

        # # Create Video using moviepy
        # for view_name in stage_scripts.keys():
        #     imageset_path = os.path.join(OUTPUT_IMAGES_DIR, view_name)
        #     imageset = [
        #         os.path.join(imageset_path, path)
        #         for path in os.listdir(imageset_path) if path[-3:] == "png"
        #     ]
        #     imageset.sort()
        #     filename = OUTPUT_VIDEO_PATH_NAME + "_" + view_name + ".mp4"

        #     FPS=3
        #     #pdb.set_trace()
        #     clip = ImageSequenceClip(imageset, fps=FPS)
        #     clip.write_videofile(filename, fps=10)


## ================================================================================
#    Draw Plot for PCL and PBD result and 3D centerline along gravity
## ================================================================================
class PCLPBD3Dcenterlinegravity():
    def DrawPCLPBDCenterline(self, index_draw_pcl, simnode_file_path,
                             savefig_PCLPBDCenterline_path, draw_pcl):

        fig_curve = plt.figure()
        ax_curve = Axes3D(fig_curve)
        #pdb.set_trace()

        savefig_file = "../DATA/PCLPBD3Dcenterline" + "/" + savefig_PCLPBDCenterline_path
        if not os.path.exists(savefig_file):
            os.makedirs(savefig_file)

        # -------- Begin to load the data from file --------
        for index_frame in index_draw_pcl:

            # -------- load pcl data and draw --------
            if draw_pcl == True:
                pcl_data = torch.as_tensor(
                    np.load(
                        "../DATA/DATA_BaxterRope/downsample_pcl_sorted/%d.npy"
                        % index_frame))
                ax_curve.scatter(pcl_data[:, 0], pcl_data[:, 1], pcl_data[:,
                                                                          2])

            # -------- load sim node data and draw--------
            sim_node_filepath = simnode_file_path
            sim_node = torch.as_tensor(
                np.load(sim_node_filepath + "/frame_%d.npy" % index_frame))
            ax_curve.scatter(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])
            ax_curve.plot3D(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])

            # -------- load centerline_3d_gravity data and draw --------
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % index_frame))
            ax_curve.scatter(
                centerline_3d_gravity[:, 0],
                centerline_3d_gravity[:, 1],
                centerline_3d_gravity[:, 2],
            )
        ax_curve.set_xlabel("X")
        ax_curve.set_ylabel("Y")
        ax_curve.set_zlabel("Z")

        # --------- Save plot Image --------
        #pdb.set_trace()
        savefig_path = savefig_file + "/" + str(index_frame) + "_.png"
        plt.savefig(savefig_path)
        plt.show()


## ================================================================================
#    Draw Plot for ALL 3d centerline and PBD result
## ================================================================================
class AllCenterlinePBCInSameImage():
    def DrawAllCenterlinePBCInSameImage(self, index_draw_pcl, loss_type_file,
                                        loss_type, png_name):

        fig_curve = plt.figure()
        ax_curve = Axes3D(fig_curve)

        savefig_file = "../DATA/AllCenterlinePBCInSameImage" + "/" + loss_type
        if not os.path.exists(savefig_file):
            os.makedirs(savefig_file)

        # -------- Begin to load the data from file --------
        for index_frame in index_draw_pcl:

            # -------- load pcl data and draw --------
            pcl_data = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/downsample_pcl_sorted/%d.npy" %
                    index_frame))
            ax_curve.scatter(pcl_data[:, 0], pcl_data[:, 1], pcl_data[:, 2])

            # -------- load sim node data and draw--------
            sim_node_filepath = "../DATA/save_loss_withConstConvergency/" + loss_type_file
            sim_node = torch.as_tensor(
                np.load(sim_node_filepath + "/%d.npy" % index_frame))
            ax_curve.scatter(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])
            ax_curve.plot3D(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])

            # -------- load centerline_3d_gravity data and draw --------
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % index_frame))
            ax_curve.scatter(centerline_3d_gravity[:, 0],
                             centerline_3d_gravity[:, 1],
                             centerline_3d_gravity[:, 2])

        # --------- Save plot Image --------
        savefig_path = savefig_file + "/" + png_name + ".png"
        plt.savefig(savefig_path)
        #plt.cla()


## ================================================================================
#    Draw Plot for ALL real and sim control points's traj in the real experiments
## ================================================================================
class RealAndSimControlPointtraj():
    def DrawRealAndSimControlPointtraj(self, index_draw_pcl, loss_type_file,
                                       loss_type, png_name):
        fig_curve = plt.figure()
        ax_curve = Axes3D(fig_curve)

        savefig_file = "../DATA/RealAndSimControlPointtraj" + "/" + loss_type
        if not os.path.exists(savefig_file):
            os.makedirs(savefig_file)

        real_control_trajpoint = []
        sim_control_trajpoint = []
        # -------- Begin to load the data from file --------
        for index_frame in index_draw_pcl:

            # -------- load sim node data--------
            sim_node_filepath = "../DATA/save_loss_withConstConvergency/" + loss_type_file
            sim_node = torch.as_tensor(
                np.load(sim_node_filepath + "/%d.npy" % index_frame))
            sim_control_trajpoint.append(sim_node[0])

            # -------- load centerline_3d_gravity data --------
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % index_frame))
            real_control_trajpoint.append(centerline_3d_gravity[0])

        sim_control_trajpoint = np.array(sim_control_trajpoint)
        real_control_trajpoint = np.array(real_control_trajpoint)

        # -------- draw simulation control point traj --------
        ax_curve.scatter(sim_control_trajpoint[:, 0],
                         sim_control_trajpoint[:, 1], sim_control_trajpoint[:,
                                                                            2])
        ax_curve.plot3D(sim_control_trajpoint[:, 0],
                        sim_control_trajpoint[:, 1], sim_control_trajpoint[:,
                                                                           2])

        # -------- draw real control point traj --------
        ax_curve.scatter(real_control_trajpoint[:, 0],
                         real_control_trajpoint[:, 1],
                         real_control_trajpoint[:, 2])
        ax_curve.plot3D(real_control_trajpoint[:, 0],
                        real_control_trajpoint[:, 1],
                        real_control_trajpoint[:, 2])

        # --------- Save plot Image --------
        savefig_path = savefig_file + "/" + png_name + ".png"
        plt.savefig(savefig_path)


## ================================================================================
#    Draw surface for the 3d centerline and PBD result
## ================================================================================
class SurfaceAndPBD():
    def DrawSurfaceAndPBD(self, index_draw_pcl, loss_type_file, loss_type,
                          png_name):

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        savefig_file = "../DATA/SurfaceAndPBD" + "/" + loss_type
        if not os.path.exists(savefig_file):
            os.makedirs(savefig_file)

        # -------- Begin to load the data from file --------
        for index_frame in index_draw_pcl:

            # -------- load sim node data--------
            sim_node_filepath = "../DATA/save_loss_withConstConvergency/" + loss_type_file
            sim_node = torch.as_tensor(
                np.load(sim_node_filepath + "/%d.npy" % index_frame))

            # -------- load centerline_3d_gravity data --------
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % index_frame))

            # -------- draw 3D centerline Surface --------
            surf = ax.plot_surface(centerline_3d_gravity[:, 0],
                                   centerline_3d_gravity[:, 1],
                                   centerline_3d_gravity[:, 2],
                                   cmap=cm.coolwarm,
                                   linewidth=0,
                                   antialiased=False)

            # -------- draw 3D centerline  --------
            ax.plot3D(centerline_3d_gravity[:, 0], centerline_3d_gravity[:, 1],
                      centerline_3d_gravity[:, 2])

            # -------- draw Sim Node  --------
            ax.plot3D(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])
            ax.scatter(sim_node[:, 0], sim_node[:, 1], sim_node[:, 2])

            fig.colorbar(surf, shrink=0.5, aspect=5)

        # --------- Save plot Image --------
        savefig_path = savefig_file + "/" + png_name + ".png"
        plt.savefig(savefig_path)


## ================================================================================
#   Plotting Responding points for the real experiments
## ================================================================================


class PlottingRespondingPoint():
    def DrawPlotingRespondingPoint(self, index_draw_pcl, loss_type_file):
        # -------- Begin to load the data from file --------
        for index_frame in index_draw_pcl:

            # -------- load sim node data--------
            sim_node_filepath = "../DATA/save_loss_withConstConvergency/" + loss_type_file
            sim_node = torch.as_tensor(
                np.load(sim_node_filepath + "/%d.npy" % index_frame))

            # -------- load centerline_3d_gravity data --------
            centerline_3d_gravity = torch.as_tensor(
                np.load(
                    "../DATA/DATA_BaxterRope/centerline_3d_gravity_clean/%d.npy"
                    % index_frame))


## ================================================================================
#   Plotting real and simulation result for the simulation experiments
## =============================================================================
class TrajAndSimNodeForSimulation():
    def DrawTrajAndSimNodeForSimulation(self, file_path_sim_node,
                                        traj_real_shape_name, index_simnodes,
                                        draw_cotroltraj):

        fig = plt.figure()
        ax_curve = Axes3D(fig)
        ax_curve.set_xlabel('X')
        #ax_curve.set_ylim3d([0, 40])
        ax_curve.set_ylabel('Y')
        ax_curve.set_zlim3d([-2, 10])
        ax_curve.set_zlabel('Z')

        #------- Load all npy files in the file for sim node result ---------
        files_sim_node = os.listdir(file_path_sim_node)

        #------- Read the data for real control point and simulation control points ---------
        sim_control_waypoints = []
        sim_target_waypoints = []
        for file_index in range(len(files_sim_node) - 1):
            sim_node = np.load(file_path_sim_node + "/" +
                               "frame_%d.npy" % file_index)
            sim_control_waypoints.append(sim_node[17, :])
            sim_target_waypoints.append(sim_node[15, :])
        sim_control_waypoints = np.array(sim_control_waypoints)
        sim_target_waypoints = np.array(sim_target_waypoints)

        # -------- Draw the plot for simulation control points --------
        ax_curve.scatter(sim_target_waypoints[:, 0],
                         sim_target_waypoints[:, 1], sim_target_waypoints[:,
                                                                          2])

        # -------- Draw the plot for real  control points --------
        traj_real_shape = np.load(file_path_sim_node + "/" +
                                  traj_real_shape_name +
                                  ".npy")[0:len(files_sim_node)]

        ax_curve.scatter(traj_real_shape[:, 0], traj_real_shape[:, 1],
                         traj_real_shape[:, 2])
        ax_curve.plot3D(traj_real_shape[:, 0], traj_real_shape[:, 1],
                        traj_real_shape[:, 2])

        if draw_cotroltraj == True:
            ax_curve.scatter(sim_control_waypoints[:, 0],
                             sim_control_waypoints[:, 1],
                             sim_control_waypoints[:, 2])
            ax_curve.plot3D(sim_control_waypoints[:, 0],
                            sim_control_waypoints[:, 1],
                            sim_control_waypoints[:, 2])

        # -------- Draw the plot for rods for specific index --------
        for index_simnode in index_simnodes:
            sim_nodes = np.load(file_path_sim_node + "/" +
                                "frame_%d.npy" % index_simnode)
            ax_curve.scatter(sim_nodes[:, 0], sim_nodes[:, 1], sim_nodes[:, 2])
            ax_curve.plot3D(sim_nodes[:, 0], sim_nodes[:, 1], sim_nodes[:, 2])
        plt.show()


## ================================================================================
#   Plotting the projection sim node from 3d and plot the 2D centerline image
## ================================================================================
class Simnode2DprojectionAnd2Dcenterline():
    def DrawSimnode2DprojectionAnd2Dcenterline(self, loss_file):

        # -------- Initial the visulization tool --------
        vistool = VisulizationTool()

        # -------- Create the file to save the plot --------
        save_fig_path = "../DATA/Simnode2DprojectionAnd2Dcenterline/" + loss_file
        if not os.path.exists(save_fig_path):
            os.makedirs(save_fig_path)
        # -------- Read All the npy files --------
        file_path = "../DATA/save_loss_withConstConvergency/" + loss_file
        npy_files = os.listdir(file_path)
        for file in npy_files:
            if os.path.splitext(file)[1] == ".npy":
                if int(os.path.splitext(file)[0].split('_')[1]) == 14:
                    continue
                frame_index = int(os.path.splitext(file)[0].split('_')
                                  [1])  #The index of the frame from npy file

                #-------- Load 2D centerline image --------
                centerline_2d_img = np.load(
                    "../DATA/DATA_BaxterRope/centerline_2d_image/" +
                    "%d.npy" % frame_index)
                plt.plot(centerline_2d_img[:, 0], centerline_2d_img[:, 1])

                #-------- Load Sim Node and projection to 2D  image --------
                sim_node = torch.as_tensor(
                    np.load("../DATA/save_loss_withConstConvergency/" +
                            loss_file + "/frame_%d.npy" % frame_index))[:, :3]
                sim_node_2D = vistool.ProjectionPtfrom3DNode(sim_node)
                plt.plot(sim_node_2D[:, 0], sim_node_2D[:, 1])

                plt.savefig(save_fig_path + "/%d.png" % frame_index)
                plt.cla()


## ================================================================================
#   Plotting the constraints situation for each
## ================================================================================
class TempConstraintDataForForward():
    def DrawStepLayer(self, num_frame, filepath, steps, layers, iteration):
        # ------ Load the files --------
        file_location = "../DATA/save_loss_withConstConvergency/" + filepath + "/temp_data_frame_layer_step_frame" + str(
            num_frame)
        C_bending = []
        C_strain = []
        C_dist = []
        loss_list = []

        # ------ Save the constraints --------
        for itr in range(iteration):
            for i in range(steps):
                for j in range(layers):
                    step_layer_simnode = np.load(file_location + "/frame_" +
                                                 str(num_frame) + "_itr_" +
                                                 str(itr) + "_step_" + str(i) +
                                                 "_layer_" + str(j) + ".npy")
                    C_dist.append(np.max(step_layer_simnode[:19, 4]))
                    C_strain.append(np.max(step_layer_simnode[:19, 5]))
                    C_bending.append(np.max(step_layer_simnode[:18, 6]))
                    loss_list.append(step_layer_simnode[19, 4])

        # ------ Plot the curve for different curves --------
        plt.plot(np.linspace(0, iteration, steps * layers * iteration),
                 C_dist,
                 label="C_dist")
        plt.plot(np.linspace(0, iteration, steps * layers * iteration),
                 C_strain,
                 label="C_strain")
        plt.plot(np.linspace(0, iteration, steps * layers * iteration),
                 C_bending,
                 label="C_bending")
        # plt.plot(np.linspace(0, iteration, steps * layers * iteration),
        #          loss_list,
        #          label="loss")
        plt.legend()
        plt.savefig(file_location + "/" + str(num_frame) + "_" +
                    str(iteration) + "_" + str(steps) + "_" + str(layers) +
                    ".png")
        plt.show()
