# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import torch
import time
import random 
import h5py
from tqdm.auto import tqdm, trange
import argparse
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()
# args_cli = parser.parse_args()

# # launch omniverse app
# config = {"headless": args_cli.headless}  
# config = {}  
simulation_app = SimulationApp({"headless": False}) #python启动时，simulationapp要立即启动（见isaac tutorial hello world）
# simulation_app.initialize(config)



# from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core import World
from omni.isaac.core.objects import FixedCuboid, FixedCylinder
# from omni.isaac.examples.base_sample import BaseSample
from omni.isaac.franka import Franka
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.types import ArticulationAction

# _isaac_sim/exts/omni.isaac.franka/omni/isaac/franka/franka.py
# from omni.isaac.orbit.markers import StaticMarker
# from omni.isaac.orbit.controllers.mpinets_controller.mpinets.data_loader import PointCloudBase



from omni.isaac.orbit.controllers.mpinets_controller.mpinets.model import MotionPolicyNetwork
from omni.isaac.orbit.controllers.mpinets_controller.mpinets.metrics import Evaluator
from omni.isaac.orbit.controllers.mpinets_controller.mpinets.geometry import construct_mixed_point_cloud
from omni.isaac.orbit.controllers.mpinets_controller.mpinets.utils import normalize_franka_joints, unnormalize_franka_joints
from omni.isaac.orbit.controllers.mpinets_controller.mpinets.types import PlanningProblem, ProblemSet
from robofin.robots import FrankaRobot, FrankaGripper
from robofin.pointcloud.torch import FrankaSampler 


# from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
# from omni.isaac.orbit.robots.single_arm import SingleArmManipulator

import pickle
from geometrout.transform import SE3
from geometrout.primitive import Cuboid, Cylinder
from typing import List, Union, Optional, Dict



# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# import torch
# torch.cuda.set_device(1)

END_EFFECTOR_FRAME = "right_gripper"
NUM_ROBOT_POINTS = 2048
NUM_OBSTACLE_POINTS = 4096
NUM_TARGET_POINTS = 128
MAX_ROLLOUT_LENGTH = 150


# # # launch omniverse app
# config = {"headless": args_cli.headless}  #新建config，设置headless这个选项，在使用simulationapp时可以选用headless模式
# simulation_app = SimulationApp(config) # 是否需要SimulationApp？？？待定！



def make_point_cloud_from_problem(
    q0: torch.Tensor,
    target: SE3,
    obstacle_points: np.ndarray,
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
    )
    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    random_obstacle_indices = np.random.choice(
        len(obstacle_points), size=NUM_OBSTACLE_POINTS, replace=False
    )
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[random_obstacle_indices, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz


def make_point_cloud_from_primitives(
    q0: torch.Tensor,
    target: SE3,
    obstacles: List[Union[Cuboid, Cylinder]],  #obstacles也是一个list
    fk_sampler: FrankaSampler,
) -> torch.Tensor:
    """
    Creates the pointcloud of the scene, including the target and the robot. When performing
    a rollout, the robot points will be replaced based on the model's prediction

    :param q0 torch.Tensor: The starting configuration (dimensions [1 x 7])
    :param target SE3: The target pose in the `right_gripper` frame
    :param obstacles List[Union[Cuboid, Cylinder]]: The obstacles in the scene
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype torch.Tensor: The pointcloud (dimensions
                         [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4])
    """
    obstacle_points = construct_mixed_point_cloud(obstacles, NUM_OBSTACLE_POINTS)
    robot_points = fk_sampler.sample(q0, NUM_ROBOT_POINTS)

    target_points = fk_sampler.sample_end_effector(
        torch.as_tensor(target.matrix).type_as(robot_points).unsqueeze(0),
        num_points=NUM_TARGET_POINTS,
    )
    xyz = torch.cat(
        (
            torch.zeros(NUM_ROBOT_POINTS, 4),
            torch.ones(NUM_OBSTACLE_POINTS, 4),
            2 * torch.ones(NUM_TARGET_POINTS, 4),
        ),
        dim=0,
    )
    xyz[:NUM_ROBOT_POINTS, :3] = robot_points.float()
    xyz[
        NUM_ROBOT_POINTS : NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS,
        :3,
    ] = torch.as_tensor(obstacle_points[:, :3]).float()
    xyz[
        NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS :,
        :3,
    ] = target_points.float()
    return xyz

def rollout_until_success(
    mdl: MotionPolicyNetwork,
    q0: np.ndarray,
    target: SE3,
    point_cloud: torch.Tensor,
    fk_sampler: FrankaSampler,
) -> np.ndarray:
    """
    Rolls out the policy until the success criteria are met. The criteria are that the
    end effector is within 1cm and 15 degrees of the target. Gives up after 150 prediction
    steps.

    :param mdl MotionPolicyNetwork: The policy
    :param q0 np.ndarray: The starting configuration (dimension [7])
    :param target SE3: The target in the `right_gripper` frame
    :param point_cloud torch.Tensor: The point cloud to be fed into the model. Should have
                                     dimensions [1 x NUM_ROBOT_POINTS + NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS x 4]
                                     and consist of the constituent points stacked in
                                     this order (robot, obstacle, target).
    :param fk_sampler FrankaSampler: A sampler that produces points on the robot's surface
    :rtype np.ndarray: The trajectory
    """
    q = torch.as_tensor(q0).unsqueeze(0).float().cuda()
    assert q.ndim == 2
    # This block is to adapt for the case where we only want to roll out a
    # single trajectory
    trajectory = [q]
    q_norm = normalize_franka_joints(q)
    assert isinstance(q_norm, torch.Tensor)
    success = False

    def sampler(config):
        return fk_sampler.sample(config, NUM_ROBOT_POINTS)

    for i in range(MAX_ROLLOUT_LENGTH):
        q_norm = torch.clamp(q_norm + mdl(point_cloud, q_norm), min=-1, max=1)
        qt = unnormalize_franka_joints(q_norm)
        assert isinstance(qt, torch.Tensor)
        trajectory.append(qt)
        eff_pose = FrankaRobot.fk(
            qt.squeeze().detach().cpu().numpy(), eff_frame="right_gripper"
        )
        # Stop when the robot gets within 1cm and 15 degrees of the target
        if (
            np.linalg.norm(eff_pose._xyz - target._xyz) < 0.01
            and np.abs(
                np.degrees((eff_pose.so3._quat * target.so3._quat.conjugate).radians)
            )
            < 15
        ):
            break
        samples = sampler(qt).type_as(point_cloud)
        point_cloud[:, : samples.shape[1], :3] = samples

    return np.asarray([t.squeeze().detach().cpu().numpy() for t in trajectory])





def calculate_metrics(mdl_path: str, problems: List[PlanningProblem]):
    mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()
    mdl.eval()
    cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
    gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
    eval = Evaluator()

    my_world = World(stage_units_in_meters=1.0)
    # franka = my_world.scene.add(Franka(prim_path="/World/Franka", name="my_franka", position=[0,0,0], orientation=[1,0,0,0]))
    my_world.scene.add_default_ground_plane() #我的似乎不用？
    my_world.reset()
    # my_franka = my_world.scene.get_object("my_franka")

    while simulation_app.is_running():
        my_world.step(render=True)
        # my_world.set_simulation_dt(physics_dt=0.5, rendering_dt=0.5) #默认为0.16666...
        # my_world.get_rendering_dt()
        # my_world.get_physics_dt()
        # print("rendering_dt: ", my_world.get_rendering_dt())
        # print("physics_dt: ", my_world.get_physics_dt())
        if my_world.is_playing():
            if my_world.current_time_step_index == 0:
                my_world.reset()

            for scene_type, scene_sets in problems.items():
                for problem_type, problem_set in scene_sets.items():
                    eval.create_new_group(f"{scene_type}, {problem_type}")
                    for problem in tqdm(problem_set, leave=False):
                        # franka = HelloWorld.setup_scene.world.scene.add #参数如何传入？setup_scene中的problem相关名称是不是要改？
                        my_world.scene.add(Franka(prim_path="/World/Franka", name="my_franka", position=[0,0,0], orientation=[1,0,0,0]))
                        my_franka = my_world.scene.get_object("my_franka")
                        my_world.reset()

                        if problem.obstacle_point_cloud is None:
                            point_cloud = make_point_cloud_from_primitives(
                                torch.as_tensor(problem.q0).unsqueeze(0),
                                problem.target,
                                problem.obstacles,
                                cpu_fk_sampler,
                            )
                        else:
                            assert len(problem.obstacles) > 0
                            point_cloud = make_point_cloud_from_problem(
                                torch.as_tensor(problem.q0).unsqueeze(0),
                                problem.target,
                                problem.obstacle_point_cloud,
                                cpu_fk_sampler,
                            )
                        start_time = time.time()
                        trajectory = rollout_until_success(
                            mdl,
                            problem.q0,
                            problem.target,
                            point_cloud.unsqueeze(0).cuda(),
                            gpu_fk_sampler,
                        )
                        eval.evaluate_trajectory(
                            trajectory,
                            0.08,  # We assume the network is to operate at roughly 12hz
                            problem.target,
                            problem.obstacles,
                            problem.target_volume,
                            problem.target_negative_volumes,
                            time.time() - start_time,
                        )
                        # print("trajectory output from mpinets: ", trajectory)
                        # print("****************************************")



                        #向场景添加target EE或target 整个robot
                        # my_franka.end_effector.  #没想好
                        # my_franka.gripper.set_joint_positions(problem.target)  #没想好

                        cube_prim_path = find_unique_string_name(
                            initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
                        )  
                        # # print(problem.target.so3)
                        # # print(type(problem.target))  #输出：class 'geometrout.transform.SE3'
                        # # print(type(problem.target.xyz))  #输出：class 'list'
                        # print(problem.target.xyz)  #输出：[0.07721892973355943, -0.4431428241168158, 0.4323637132137983]
                        # print(problem.target.so3.wxyz) #输出：[-0.12440869437591286, 0.7126188027223423, -0.6902976972437753, -0.013638473162225299]
                      
                        c = random.randint(1,10000)
                        my_world.scene.add(
                            FixedCuboid(
                                # prim_path="/World/cuboid"+str(i),
                                # prim_path="/home/omniverse/orbit/_isaac_sim/exts/omni.isaac.core/omni/isaac/core/World/random_cuboid",
                                prim_path=cube_prim_path,
                                # prim_path="/World/Cube",
                                name="target_cuboid"+str(c),
                                position=problem.target.xyz,
                                # position=problem.target_volume.center,
                                orientation=problem.target.so3.wxyz,
                                # orientation=problem.target_volume.quaternion, 
                                # scale=np.array([0.01, 0.01, 0.01]), 
                                size=0.01,
                                color=np.array([1.0, 0, 0]),
                                # size=problem.obstacles.cuboid_size[???], #size这行暂时先这样写着，肯定不对，dims没弄明白具体指什么？？
                            )   
                        )                             

                        #向场景添加obstacles
                        # 以下是使用item字典写的（但不知道items和item关系？暂时先写的items）。
                        # 使用item或items写对吗？还是说可以使用obstacles写：for o in obstacles, if cylinder,...; if cuboid,...。??
                        


                        # cuboid_name = find_unique_string_name(
                        #     initial_name="my_cuboid", is_unique_fn=lambda x: not my_world.scene.object_exists(x)
                        # )
                        # cylinder_name = find_unique_string_name(
                        #     initial_name="my_cylinder", is_unique_fn=lambda x: not my_world.scene.object_exists(x)
                        # )
                        # i = 0
                        # print("total number of obstacles: ", len(problem.obstacles))
                        A = []
                        B = []
                        for o in problem.obstacles:
                            if isinstance(o, Cuboid):
                                # print("this is a cuboid.")
                                a = random.randint(1,100000)
                                cube_prim_path = find_unique_string_name(
                                    initial_name="/World/Cube", is_unique_fn=lambda x: not is_prim_path_valid(x)
                                )
                                
                                # print(o.center)  #已验证：可以输出center
                                # print(o.center[0])  #已验证：可以输出center的第一个值
                                # print(type(o.center))  #已验证：center类型为list
                                # print(o.pose)
                                # print(type(o.pose))  #输出：class 'geometrout.transform.SE3'
                                # print(o.pose.xyz)
                                # print(o.pose.so3)
                                # print(type(o.pose.so3)) #输出：class 'geometrout.transform.SO3'
                                # print(o.pose.so3.wxyz)  #输出：[1.0, 0.0, 0.0, 0.0]
                                my_world.scene.add(
                                    FixedCuboid(
                                        # prim_path="/World/cuboid"+str(i),
                                        # prim_path="/home/omniverse/orbit/_isaac_sim/exts/omni.isaac.core/omni/isaac/core/World/random_cuboid",
                                        prim_path=cube_prim_path,
                                        # prim_path="/World/cuboid",
                                        # name="fancy_cuboid123"+str(i),
                                        # name="fancy_cuboid"+str(random.randint(1,100000)),
                                        name="fancy_cuboid"+str(a),
                                        # name=cuboid_name,
                                        # position=o.center,
                                        position=o.pose.xyz,
                                        orientation=o.pose.so3.wxyz, #报错：'Cuboid' object has no attribute 'quaternion'
                                        scale=np.array([1, o.dims[1]/o.dims[0], o.dims[2]/o.dims[0]]), 
                                        size=o.dims[0], #不知道对不对 ---对
                                        color=np.array([0.1, 0.5, 0.3]),

                                        # size=problem.obstacles.cuboid_size[???], #size这行暂时先这样写着，肯定不对，dims没弄明白具体指什么？？
                                    )
                                )
                                # print("random number of the cuboid is: ", a)
                                A.append(str(a))
                                # print("a cuboid is added!")
                                # i +=1
                                
                            elif isinstance(o, Cylinder):
                                # print("this is a cylinder.")
                                b = random.randint(1,100000)
                                cylinder_prim_path = find_unique_string_name(
                                    initial_name="/World/Cylinder", is_unique_fn=lambda x: not is_prim_path_valid(x)
                                )                                  
                                my_world.scene.add(
                                    FixedCylinder(
                                        # prim_path="/World/cylinder"+str(i),
                                        # prim_path="/home/omniverse/orbit/_isaac_sim/exts/omni.isaac.core/omni/isaac/core/World/random_cylinder",
                                        prim_path=cylinder_prim_path,
                                        # prim_path="/World/cylinder",
                                        # name="fancy_cylinder"+str(i),
                                        # name="fancy_cylinder_"+str(random.randint(1,100000)),
                                        name="fancy_cylinder"+str(b),
                                        # name=cylinder_name,
                                        radius=o.radius,
                                        height=o.height,
                                        position=o.pose.xyz,
                                        orientation=o.pose.so3.wxyz, #应该怎么写
                                        scale=np.array([1, 1, 1]),
                                        color=np.array([0.1, 0.7, 0.3]),
                                    )
                                )
                                # print("random number of the cylinder is:", b)
                                B.append(str(b))
                                # print("a cylinder is added!")
                                # i += 1
                        # print("list A is: ",A)
                        # print("list B is: ",B)



                        #——————以下代码可以跑通没有问题————————
                        # trajectory为由多个q（1*7的numpy数组）组成的numpy数组，但apply_action需要1*9的关节位置控制命令。
                        # 因此，下面两行是将trajectory变成多个（1*9）numpy数组组成的numpy数组（在后面补两个0，即：机械爪先暂时设置为0）.
                        trajectory_9d = [np.hstack((arr, np.ones(2))) for arr in trajectory]
                        trajectory_9d = np.array(trajectory_9d)
                        # print("trajectory_9d: ",trajectory_9d)

                        #依次渲染每个路径点位姿（每个渲染帧都使用apply_action执行机械臂关节空间位置控制）
                        for q in trajectory_9d:
                            # print("q: ",q) #q取值没有问题，与trajectory_9d每行完全一致
                            # print("**********")

                            my_franka.apply_action(ArticulationAction(joint_positions=q))
                            # print("applied_action: ", my_franka.get_applied_action()) #输出显示applied action与q完全相同。到底为什么与原mpinets走的路径不一样呢？
                            # my_franka.get_joint_state()
                            # print("**********")


                            
                            time.sleep(0.1)
                            my_world.step(render=True)
                        #————————以上代码可以跑通没有问题————————
 
                            

                        #在这里加需要步进渲染的东西，估计就不用复杂的来回调用了。
                        #先读arg，然后class，调用calculate（return）即可。

                        #还需要得到的trajectory中的每个q，都进行步进渲染robot。
                        #渲染初始位姿
                        # my_franka.set_world_pose(position=problem.q0[:3], orientation=problem.q0[3:])
                        # time.sleep(0.2)
                        #依次渲染每个路径点位姿
                        # for q in trajectory:
                        #     my_franka.set_world_pose(position=q[:3], orientation=q[3:])
                        #     time.sleep(0.2)
                        #     my_world.step(render=True)
                        
                        
                        # my_world.scene.remove_object(str.startswith("fancy_cuboid"))
                        # my_world.scene.remove_object(str.startswith("fancy_cylinder"))
                        # my_world.scene.remove_object(str.startswith("target_cuboid"))
                        # my_world.clear()  #不知道哪个清理的程度合适 先试一下这个————————不可！会使整个程序结束，终端显示Segmentation fault (core dumped)。
                        # my_world.scene.remove_object(registry_only=True)  #试一下这个
                        # my_world.scene.remove_object(name="fancy_cuboid"+any())#再试一下这个
                        # my_world.scene.remove_object(name=find_unique_string_name("fancy_cuboid",is_unique_fn=lambda x: not is_prim_path_valid(x))) #试一下这个

                        #remove cuboid obstacles
                        for i in range(len(A)):
                            my_world.scene.remove_object(name="fancy_cuboid"+str(A[i-1]),registry_only=False)
                            # print("removed"+str(A[i-1]))
                        # print("cuboid obstacles are removed")

                        # remove cylinder obstacles
                        for j in range(len(B)):
                            my_world.scene.remove_object(name="fancy_cylinder"+str(B[j-1]),registry_only=False)
                            # print("removed"+str(A[j-1]))
                        # print("cylinder obstacles are removed")
                        
                        # remove target cuboid
                        my_world.scene.remove_object(name="target_cuboid"+str(c),registry_only=False)
                        # print("target are removed")

                        # remove robot
                        my_world.scene.remove_object(name="my_franka",registry_only=False)
                        # print("robot are removed")


                        time.sleep(2)


                        # for b in B:
                        #     my_world.scene.remove_object(name="fancy_cylinder"+b,registry_only=False)
                        #     print("removed"+b)
                        # print("cylinder obsracles are removed")
                        # my_world.clear()

                    print(f"Metrics for {scene_type}, {problem_type}")
                    eval.print_group_metrics()
                    simulation_app.close()
            # print("Overall Metrics")
            # eval.print_overall_metrics()

                        # for Cuboid in problem.obstacles:
                        #     position=problem.obstacles
                        #     my_world.scene.add(
                        #         DynamicCuboid(
                        #             prim_path="/World/random_cuboid",
                        #             name="fancy_cuboid",
                        #             # position=problem.obstacles"Cu",
                        #             # orientation=problem.obstacles.quaternion,
                        #             # size=problem.obstacles.cuboid_size[???], #size这行暂时先这样写着，肯定不对，dims没弄明白具体指什么？？
                        #         )
                        #     )
                        # for Cylinder in problem.obstacles:
                        #     my_world.scene.add(
                        #         DynamicCylinder(
                        #             prim_path="/World/random_cylinder",
                        #             name="fancy_cylinder",
                        #             radius=problem.obstacles.radius,
                        #             height=problem.obstacles.height,
                        #             position=problem.obstacles.center,
                        #             orientation=problem.obstacles.quaternion,
                        #         )
                        #     )

                        # cylinder_radii = problem.obstacles["cylinder_radii"]
                        # cylinder_height = problem.items["cylinder_heights"]
                        # cylinder_position = problem.items["cylinder_centers"]
                        # cylinder_orientation = problem.items["cylinder_quats"]
                        # n = list(problem.obstacles).count(Cuboid)  #统计当前这个problem列表中，Cuboid的个数。若此方法不行，可尝试下一行法二
                        # # n = list(problem[3]).count("Cuboid")  #法二：列表problem为PlanningProblem,其第3个元素为obstacles
                        # cuboid_size = problem.items["cuboid_dims"]
                        # cuboid_position = problem.items["cuboid_centers"]
                        # cuboid_orientation = problem.items["cuboid_quats"]
                        # m = problem.items["cuboid_centers"].size(0)
                        # for i in range(n-1):
                        #     World.scene.add(
                        #         DynamicCylinder(
                        #             prim_path="/World/random_cylinder",
                        #             name="fancy_cylinder",
                        #             radius=cylinder_radii[i],
                        #             height=cylinder_height[i],
                        #             position=cylinder_position[i],
                        #             orientation=cylinder_orientation[i],
                        #         )
                        #     )
                        # for j in range(m-1):
                        #     World.scene.add(
                        #         DynamicCuboid(
                        #             prim_path="/World/random_cuboid",
                        #             name="fancy_cuboid",
                        #             position=center,
                        #             orientation=quaternion,
                        #             size=cuboid_size[???], #size这行暂时先这样写着，肯定不对，dims没弄明白具体指什么？？
                        #         )
                        #     )
                            

# def change():
#     return ArticulationAction(joint_positions=trajectory)
    



# 现在还有疑问：关于执行步骤的。上面line209左右的那些循环，我想让加载场景与其中某循环同步，应该怎么做？？？


        #向场景添加franka 原mpinets世界和robot都不重新加载，只刷新掉obstacles，target，start


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
    "mdl_path", type=str, help="A checkpoint file from training MotionPolicyNetwork"
    )
    parser.add_argument(
        "problems",
        type=str,
        help="A pickle file of sample problems that follow the PlanningProblem format",
    )
    parser.add_argument(
        "environment_type",
        choices=["tabletop", "cubby", "merged-cubby", "dresser", "all"],
        help="The environment class",
    )
    parser.add_argument(
        "problem_type",
        choices=["task-oriented", "neutral-start", "neutral-goal", "all"],
        help="The type of planning problem",
    )
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help=(
            "If set, uses a partial view pointcloud rendered in Pybullet. If not set,"
            " uses pointclouds sampled from every side of the primitives in the scene"
        ),
    )
    parser.add_argument(
        "--skip-visuals",
        action="store_true",
        help=(
            "If set, will not show visuals and will only display metrics. This will be"
            " much faster because the trajectories are not displayed"
        ),
    )
    args = parser.parse_args()
    with open(args.problems, "rb") as f:
        problems = pickle.load(f)
    env_type = args.environment_type.replace("-", "_")
    problem_type = args.problem_type.replace("-", "_")
    if env_type != "all":
        problems = {env_type: problems[env_type]}
    if problem_type != "all":
        for k in problems.keys():
            problems[k] = {problem_type: problems[k][problem_type]}

# if args.use_depth:
#     convert_primitive_problems_to_depth(problems)
    if args.skip_visuals:
        calculate_metrics(args.mdl_path, problems)
# else:
    # visualize_results(args.mdl_path, problems)
    # visualize_results(args.mdl_path, problems)
    # calculate_metrics(args.mdl_path, problems)
    # Close the simulator
    # simulation_app.close()


        




    


        #向场景添加obstacles：
        #读取pickle文件的相应信息:
        #创建一个空列表item，读取pickle文件中的以下我所需的关于长方体的抬头的数据，统一成2维数组
        # 疑问： trajectory_idx是从哪里获得的？？？
"""
        with h5py.File(str(PointCloudBase._database), "r") as f:
            item = {}
            cuboid_dims = f["cuboid_dims"][trajectory_idx, ...]
            if cuboid_dims.ndim == 1:
                cuboid_dims = np.expand_dims(cuboid_dims, axis=0)

            cuboid_centers = f["cuboid_centers"][trajectory_idx, ...]
            if cuboid_centers.ndim == 1:
                cuboid_centers = np.expand_dims(cuboid_centers, axis=0)

            cuboid_quats = f["cuboid_quaternions"][trajectory_idx, ...]
            if cuboid_quats.ndim == 1:
                cuboid_quats = np.expand_dims(cuboid_quats, axis=0)
            # Entries without a shape are stored with an invalid quaternion of all zeros
            # This will cause NaNs later in the pipeline. It's best to set these to unit
            # quaternions.
            # To find invalid shapes, we just look for a dimension with size 0
            cuboid_quats[np.all(np.isclose(cuboid_quats, 0), axis=1), 0] = 1

            # 读到的数据写入item列表
            # Leaving in the zero volume cuboids to conform to a standard
            # Pytorch array size. These have to be filtered out later
            item["cuboid_dims"] = torch.as_tensor(cuboid_dims)
            item["cuboid_centers"] = torch.as_tensor(cuboid_centers)
            item["cuboid_quats"] = torch.as_tensor(cuboid_quats)

            #同样地，读取以下关于圆柱的抬头的数据，写入item列表
            if "cylinder_radii" not in f.keys():
                # Create a dummy cylinder if cylinders aren't in the hdf5 file
                cylinder_radii = np.array([[0.0]])
                cylinder_heights = np.array([[0.0]])
                cylinder_centers = np.array([[0.0, 0.0, 0.0]])
                cylinder_quats = np.array([[1.0, 0.0, 0.0, 0.0]])
            else:
                cylinder_radii = f["cylinder_radii"][trajectory_idx, ...]
                if cylinder_radii.ndim == 1:
                    cylinder_radii = np.expand_dims(cylinder_radii, axis=0) #变成二维数组
                cylinder_heights = f["cylinder_heights"][trajectory_idx, ...]
                if cylinder_heights.ndim == 1:
                    cylinder_heights = np.expand_dims(cylinder_heights, axis=0)
                cylinder_centers = f["cylinder_centers"][trajectory_idx, ...]
                if cylinder_centers.ndim == 1:
                    cylinder_centers = np.expand_dims(cylinder_centers, axis=0)
                cylinder_quats = f["cylinder_quaternions"][trajectory_idx, ...]
                if cylinder_quats.ndim == 1:
                    cylinder_quats = np.expand_dims(cylinder_quats, axis=0)
                # Ditto to the comment above about fixing ill-formed quaternions
                cylinder_quats[np.all(np.isclose(cylinder_quats, 0), axis=1), 0] = 1

            item["cylinder_radii"] = torch.as_tensor(cylinder_radii)
            item["cylinder_heights"] = torch.as_tensor(cylinder_heights)
            item["cylinder_centers"] = torch.as_tensor(cylinder_centers)
            item["cylinder_quats"] = torch.as_tensor(cylinder_quats)
            """
        #现在item为字典，键与值对应，我现在要让isaac sim去查item字典，获取障碍物信息


        # #创建名为cuboids的列表，它是：名为Cuboid的数组组成的列表。每个Cuboid数组的三个元素分别从其对应的三个列表中得到数据
        # cuboids = [
        #     Cuboid(c, d, q)
        #     for c, d, q in zip(
        #         list(cuboid_centers), list(cuboid_dims), list(cuboid_quats)
        #     )
        # ]

        # # Filter out the cuboids with zero volume
        # cuboids = [c for c in cuboids if not c.is_zero_volume()]

        # # 同样地，创建名为cylinders的列表。
        # cylinders = [
        #     Cylinder(c, r, h, q)
        #     for c, r, h, q in zip(
        #         list(cylinder_centers),
        #         list(cylinder_radii.squeeze(1)),
        #         list(cylinder_heights.squeeze(1)),
        #         list(cylinder_quats),
        #     )
        # ]
        # cylinders = [c for c in cylinders if not c.is_zero_volume()]
        
        # # 循环遍历cuboids列表，来向world中添加物体  
        # # 疑问：DynamicCuboid的参数为size，mpinets给定的随机尺寸为长短边，不知道怎么转换！
        # # 疑问：读取列表的数据那些计数[]对吗？
        # for i, cuboid in enumerate(cuboids):
        #     CUBOIDS[i] = world.scene.add(
        #         DynamicCuboid(
        #             prim_path = "/World/random_cube",
        #             name = "cuboid"[i],
        #             position = cuboid[i][0],
        #             size = np.array([???]),
        #             orientation = cuboids[i][2], 
        #         )
        #     )
        # # 循环遍历cylinders列表，来向world中添加物体
        # # 疑问：读取列表的数据那些计数[]对吗？
        # for i, cylinder in enumerate(cylinders):
        #     CYLINDERS[i] = world.scene.add(
        #         DynamicCylinder(
        #         prim_path = "/World/random_cylinder",
        #         name = "fancy_cylinder"[i],
        #         position = cylinder[i][0],
        #         radius = cylinder[i][1],
        #         height = cylinder[i][2],
        #         orientation = cylinder[i][3],
        #         )
        #     )
        







# @torch.no_grad()
# def visualize_results(mdl_path: str, problems: ProblemSet):
#     """
#     Runs a sequence of problems and visualizes the results in Pybullet

#     :param mdl_path str: The path to the model
#     :param problems List[PlanningProblem]: A list of problems
#     """

#     mdl = MotionPolicyNetwork.load_from_checkpoint(mdl_path).cuda()  #mdl_path：.ckpt文件输入（是训练好的模型权重）
#     mdl.eval()   #eval定义在哪里？什么用？
#     cpu_fk_sampler = FrankaSampler("cpu", use_cache=True)
#     gpu_fk_sampler = FrankaSampler("cuda:0", use_cache=True)
#     # Load kit helper
#     sim = SimulationContext(physics_dt=0.01, rendering_dt=0.0-1, backend="torch", device="cuda:0")
#     eval = Evaluator()
    
#     # Enable GPU pipeline and flatcache 
#     if sim.get_physics_context().use_gpu_pipeline:
#         sim.get_physics_context().enable_flatcache(True)
#     # Enable hydra scene-graph instancing
#     set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
    


#     # obstacles（包括table/clear_table, objects<cubiods,cubic>）pybullet渲染出的这些东西都是同一个颜色，它只知道是分别不同的东西，但不分是桌子还是桌上的障碍物。

    





#     #在每加入assets之后都建议reset world
#     world.reset()



#     # Spawn things into stage
#     # Markers
#     ee_marker = StaticMarker("/Visuals/ee_current", count=1, scale=(0.1, 0.1, 0.1))
#     goal_marker = StaticMarker("/Visuals/ee_goal", count=1, scale=(0.1, 0.1, 0.1))
    
    
#     robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
#     # spawn robot
#     robot = SingleArmManipulator(cfg=robot_cfg)
#     robot.spawn("/World/envs/env_0/Robot", translation=(0.0, 0.0, 0.0))

#     # 环境场景需要在这里设置还是在循环中？ik示例是在这里设置，mpinets是在循环中。
    
    
    
#     # Play the simulator（在放置好东西之后reset）
#     sim.reset()
#     # Acquire handles
#     # Initialize handles
#     robot.initialize("/World/envs/env_.*/Robot")
#     # Reset states
#     robot.reset_buffers()

#     # Now we are ready!
#     print("[INFO]: Setup complete...")

#     # Create buffers to store actions
#     # ik_commands = torch.zeros(robot.count, ik_controller.num_actions, device=robot.device)
#     robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)


#     # Define simulation stepping
#     sim_dt = sim.get_physics_dt()
#     # episode counter
#     sim_time = 0.0
#     count = 0
#     # Note: We need to update buffers before the first step for the controller.
#     robot.update_buffers(sim_dt)

#     # Simulate physics
#     while simulation_app.is_running():
#         # If simulation is stopped, then exit.
#         if sim.is_stopped():
#             break
#         # If simulation is paused, then skip.
#         if not sim.is_playing():
#             sim.step()
#             continue
#         # reset
#         if count % 150 == 0:
#             # reset time
#             count = 0
#             sim_time = 0.0
#             # reset dof state 
#             dof_pos, dof_vel = robot.get_default_dof_state()
#             robot.set_dof_state(dof_pos, dof_vel)
#             robot.reset_buffers()


#         #对于每个场景的每个problem：制作点云，计算，评估路径，设置点云颜色，使用meshcat可视化点云，
#         for scene_type, scene_sets in problems.items():
#             for problem_type, problem_set in scene_sets.items():
#                 for problem in tqdm(problem_set, leave=False):
#                     eval.create_new_group(f"{scene_type}, {problem_type}")
#                     if problem.obstacle_point_cloud is None:
#                         point_cloud = make_point_cloud_from_primitives(
#                             torch.as_tensor(problem.q0).unsqueeze(0),
#                             problem.target,
#                             problem.obstacles,
#                             cpu_fk_sampler,
#                         )
#                     else:
#                         point_cloud = make_point_cloud_from_problem(
#                             torch.as_tensor(problem.q0).unsqueeze(0),
#                             problem.target,
#                             problem.obstacle_point_cloud,
#                             cpu_fk_sampler,
#                         )
#                     start_time = time.time() #开始计时
#                     trajectory = rollout_until_success(
#                         mdl,
#                         problem.q0,
#                         problem.target,
#                         point_cloud.unsqueeze(0).cuda(),
#                         gpu_fk_sampler,
#                     )
#                     if problem.obstacles is not None:
#                         eval.evaluate_trajectory(
#                             trajectory,
#                             0.08,  # We assume the network is to operate at roughly 12hz
#                             problem.target,
#                             problem.obstacles,
#                             problem.target_volume,
#                             problem.target_negative_volumes,
#                             time.time() - start_time,
#                         )
                    


#                     #下面这些点云可视化可以不需要！
#                     # point_cloud_colors = np.zeros(
#                     #     (3, NUM_OBSTACLE_POINTS + NUM_TARGET_POINTS)
#                     # )
#                     # point_cloud_colors[1, :NUM_OBSTACLE_POINTS] = 1
#                     # point_cloud_colors[0, NUM_OBSTACLE_POINTS:] = 1
                    
#                     #应该是使用meshcat可视化点云（mpinets中的）（isaac sim中如何可视化点云？但好像不用可视化点云也可以，只要能可视化机械臂过程即可）
#                     # viz["point_cloud"].set_object(
#                     #     # Don't visualize robot points
#                     #     meshcat.geometry.PointCloud(
#                     #         position=point_cloud[NUM_ROBOT_POINTS:, :3].numpy().T,
#                     #         color=point_cloud_colors,
#                     #         size=0.005,
#                     #     )
#                     # )
#                     # if problem.obstacles is not None:
#                     #     sim.load_primitives(problem.obstacles, visual_only=True)
                    

#                     # # in some cases the zero action correspond to offset in actuators
#                     # # so we need to subtract these over here so that they can be added later on
#                     # arm_command_offset = robot.data.actuator_pos_offset[:, : robot.arm_num_dof]
#                     # # offset actuator command with position offsets
#                     # # note: valid only when doing position control of the robot
#                     # robot_actions[:, : robot.arm_num_dof] -= arm_command_offset
#                     # # apply actions
#                     # robot.apply_action(robot_actions)
#                     # # perform step
#                     # sim.step()
#                     # # update sim-time
#                     # sim_time += sim_dt
#                     # count += 1
#                     # # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
#                     # if sim.is_playing():
#                     #     # update buffers
#                     #     robot.update_buffers(sim_dt)
#                     #     # update marker positions
#                     #     ee_marker.set_world_poses(robot.data.ee_state_w[:, 0:3], robot.data.ee_state_w[:, 3:7])
#                     #     goal_marker.set_world_poses(ik_commands[:, 0:3] + envs_positions, ik_commands[:, 3:7])



#                     # gripper.marionette(problem.target)  #应该是类似于apply action，但我似乎不需要gripper规划，只需link7末端到达target即可。如何改？？？
#                     # franka.marionette(trajectory[0])  #来自mpinets
#                     robot.set_dof_state(trajectory[0])
#                     time.sleep(0.2)
#                     for q in trajectory:
#                         # franka.control_position(q)
#                         robot.set_dof_state(trajectory[q])  #这个函数到底什么意思？再查查！想用类似于apply_action的函数，但是否需要1*7描述形式与SE3描述形式或是其他描述形式的转换？？？有点不太明白。
#                         # perform step
#                         sim.step()
#                         # update sim-time
#                         sim_time += sim_dt
#                         count += 1
#                         if sim.is_playing():
#                             # update buffers
#                             robot.update_buffers(sim_dt)
#                             # update marker positions
#                             ee_marker.set_world_poses(robot.data.ee_state_w[:, 0:3], robot.data.ee_state_w[:, 3:7])
#                             goal_marker.set_world_poses(problem.target[:, 0:3], problem.target[:, 3:7])
#                         time.sleep(0.08)
#                     sim.clear_all_obstacles()
#                 print(f"Metrics for {scene_type}, {problem_type}")
#                 eval.print_group_metrics()
#         print("Overall Metrics")
#         eval.print_overall_metrics()

                        # sim_config, _ = franka.get_joint_states()
                        # # Move meshes in meshcat to match PyBullet
                        # for idx, (k, v) in enumerate(
                        #     urdf.visual_trimesh_fk(sim_config[:8]).items()
                        # ):
                        #     viz[f"robot/{idx}"].set_transform(v)
                        
                    # Adding extra timesteps with no new controls to allow the simulation to
                    # converge to the final timestep's target and give the viewer time to look at
                    # it
                    # for _ in range(20):
                    #     sim.step()
                    #     # sim_config, _ = franka.get_joint_states()
                    #     sim_config, _ = robot.get_joint_states() #robot是否有get_joint_states相关的函数？找！（今天暂且先这么写着）
                    #     # # Move meshes in meshcat to match PyBullet
                    #     # for idx, (k, v) in enumerate(
                    #     #     urdf.visual_trimesh_fk(sim_config[:8]).items()
                    #     # ):
                    #     #     viz[f"robot/{idx}"].set_transform(v)
                    #     time.sleep(0.08)
                    


