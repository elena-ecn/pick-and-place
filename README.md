# Pick and Place

A pick-and-place application with the Franka Panda robot simulated in ROS and Gazebo. 

<p align="center" width="100%">
    <img src="images/pick_and_place_overview.png" width="600">
</p>



## Description

This project showcases a pick-and-place application that utilizes:
* A state machine
* Vision to detect objects based on their color via a Kinect RGB-D sensor

<p align="center" width="100%">
    <img src="images/pick_and_place.gif" width="600">
</p>

The state machine graph is depicted below:

<p align="center" width="100%">
    <img src="images/state_machine.png" width="500">
</p>

The robot starts in the home position. If objects are detected on the workbench,
it randomly selects one. Once an object is selected, the robot picks and places it to 
the same color bin. Then, returns to the home position. If no objects are 
detected on the workbench, it stops.



## Dependencies

The project was developed on Ubuntu 20.04 LTS with:
* ROS Noetic
* Gazebo 11.11.0

The following dependencies need to be installed:
* [Franka ROS Interface](https://github.com/justagist/franka_ros_interface/tree/master)
* [Panda Simulator](https://github.com/justagist/panda_simulator)
* [MoveIt](https://moveit.ros.org/install/)


The python dependencies can be installed from the `requirements.txt` file.
```shell
pip install -r requirements.txt
```

To resolve issues with grasping in Gazebo, the [Grasp Fix Plugin](https://github.com/JenniferBuehler/gazebo-pkgs) is utilized.



## Usage

Load the robot in Gazebo
```shell
roslaunch pick_and_place panda_world.launch 
```

Start MoveIt for motion planning
```shell
roslaunch panda_sim_moveit sim_move_group.launch
```

Run the object detector
```shell
rosrun pick_and_place object_detector.py
```

Run the pick-and-place controller
```shell
rosrun pick_and_place pick_and_place_state_machine.py
```



## License
The contents of this repository are covered under the [MIT License](LICENSE).
