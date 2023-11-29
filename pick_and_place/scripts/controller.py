#!/usr/bin/env python3

"""
Controls the robotic arm for pick-and-place tasks.

Author: Elena Oikonomou
Date:   Fall 2023
"""

import numpy as np
import random
import moveit_commander
import rospy

from franka_interface import GripperInterface
from geometry_msgs.msg import PoseStamped
from panda_robot import PandaArm
from typing import Tuple
from tf.transformations import quaternion_from_euler

from pick_and_place.msg import DetectedObjectsStamped, DetectedObject


class Controller:
    def __init__(self):

        self.red_bin = (-0.5, -0.25)        # Position (x,y) wrt world frame of red bin
        self.green_bin = (-0.5, 0.0)        # Position (x,y) wrt world frame of green bin
        self.blue_bin = (-0.5, 0.25)        # Position (x,y) wrt world frame of blue bin

        self.workbench_height = 0.2
        self.x_offset = 0.01                # Gripper offset in x-axis
        self.z_offset = 0.105               # Gripper offset in z-axis
        self.z_pre_pick_offset = 0.2        # Offset of the pre-pick position wrt the pick position in z-axis
        self.z_pre_place_offset = 0.2       # Offset of the pre-place position wrt the place position in z-axis

        self.objects_on_workbench = []      # Objects on top of the workbench
        
        self.panda = PandaArm()                                                              
        self.gripper_interface = GripperInterface()    
        self.scene_interface = moveit_commander.PlanningSceneInterface()
        self.add_collision_objects()

        rospy.Subscriber('/object_detection', DetectedObjectsStamped, self.update_objects_callback)

    def update_objects_callback(self, msg: DetectedObjectsStamped) -> None:
        """Updates the objects that are currently on top of the workbench."""
        self.objects_on_workbench = msg.detected_objects

    def select_random_object(self) -> DetectedObject:
        """Selects an object at random from the ones that are currently on top of the workbench."""
        return random.choice(self.objects_on_workbench)
    
    def are_objects_on_workbench(self) -> bool:
        """Checks whether there are any objects on top of the workbench."""
        return len(self.objects_on_workbench) > 0

    def move_object(self, object: DetectedObject) -> None:
        """Picks the given object and places it to the correct color bin."""
        x = object.x_world 
        y = object.y_world
        z = object.height + self.workbench_height 
        color = object.color
        print("\nSelected Object: {}    (x,y) = ({:.3f}, {:.3f})\n".format(color, x, y))

        self.pick(x=x, y=y, z=z)
        bin = self.select_bin(color)
        self.place(x=bin[0], y=bin[1], z=0.5)

    def select_bin(self, color: str) -> Tuple[float]:
        """Returns the (x,y) position in the wolrd frame of the given color bin."""
        if color == "red":
            bin = self.red_bin
        elif color == "green":
            bin = self.green_bin
        elif color == "blue":
            bin = self.blue_bin
        else:
            rospy.loginfo('The object color does not match an available bin color. It will be placed on the green bin.')
            bin = self.green_bin
        
        return bin

    def pick(self, x: float, y: float, z: float, roll: float=0, pitch: float=np.pi, yaw: float=0, object_width: float=0.025) -> None:
        """Picks up the object at the given position with the given end-effector orientation."""

        # Define poses
        pre_pick_position = np.array([x + self.x_offset, y, z + self.z_offset + self.z_pre_pick_offset]) 
        pick_position = np.array([x + self.x_offset, y, z + self.z_offset]) 
        pick_orientation = quaternion_from_euler(roll, pitch, yaw)

        # Pre-pick
        self.panda.move_to_cartesian_pose(pos=pre_pick_position, ori=pick_orientation)
        self.gripper_interface.open()

        # Pick
        self.panda.move_to_cartesian_pose(pos=pick_position, ori=pick_orientation)
        result = self.gripper_interface.grasp(width=object_width, force=5, speed=None, epsilon_inner=0.005, epsilon_outer=0.005)  # Grasp
        if result is not True:
            print("Warning! Grasping did not succeed.")

        # Post-pick 
        self.panda.move_to_cartesian_pose(pos=pre_pick_position, ori=pick_orientation)
    
    def place(self, x: float, y: float, z: float, roll: float=0, pitch: float=np.pi, yaw: float=0) -> None:
        """Places the object at the given position with the given end-effector orientation."""

        # Define poses
        pre_place_position = np.array([x, y, z + self.z_pre_place_offset])
        place_position = np.array([x, y, z])
        place_orientation = quaternion_from_euler(roll, pitch, yaw)

        # Pre-place
        self.panda.move_to_cartesian_pose(pos=pre_place_position, ori=place_orientation)

        # Place
        self.panda.move_to_cartesian_pose(pos=place_position, ori=place_orientation)
        self.gripper_interface.open()

        # Post-place
        self.panda.move_to_cartesian_pose(pos=pre_place_position, ori=place_orientation)

    def add_collision_objects(self) -> None:
        """Adds objects in the scene so the MoveIt planner can avoid collisions."""

        # Add workbench
        workbench_pose = PoseStamped()
        workbench_pose.header.frame_id = "world"
        workbench_pose.pose.position.x = 0.7
        workbench_pose.pose.position.y = 0.0
        workbench_pose.pose.position.z = 0.1
        workbench_size = (1.0, 3.0, 0.2)  # Box size
        self.scene_interface.add_box(name="workbench", pose=workbench_pose, size=workbench_size)

        # Add bin bench
        bin_bench_pose = PoseStamped()
        bin_bench_pose.header.frame_id = "world"
        bin_bench_pose.pose.position.x = -0.55
        bin_bench_pose.pose.position.y = 0.0
        bin_bench_pose.pose.position.z = 0.1 
        binbench_size = (0.4, 1.5, 0.2)  # Box size  (z = 0.1 + 0.1 (underbin height + bin height))
        self.scene_interface.add_box(name="binbench", pose=bin_bench_pose, size=binbench_size)

        self.wait_for_objects()

    def wait_for_objects(self, timeout: int=5) -> None:
        """Checks whether objects have been added to the scene."""
        start = rospy.get_time()
        elapsed_time = 0
        are_objects_known = False
        while elapsed_time < timeout and not rospy.is_shutdown():
            known_objects = self.scene_interface.get_known_object_names()
            are_objects_known = ("workbench" in known_objects) and ("binbench" in known_objects)

            if are_objects_known:
                return

            rospy.sleep(0.1)
            elapsed_time = rospy.get_time() - start

        print("Warning! Collision objects not yet visible in the planning scene.")
