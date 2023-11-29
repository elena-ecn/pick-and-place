#!/usr/bin/env python3

"""
Detects objects based on their color and provides information 
about their position, dimensions and color.

Author: Elena Oikonomou
Date:   Fall 2023
"""

import numpy as np
import cv2, cv_bridge
import rospy
import tf.transformations

from gazebo_msgs.srv import GetModelState
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import Image, CameraInfo
from typing import List, Tuple

from pick_and_place.msg import DetectedObjectsStamped, DetectedObject


class VisionObjectDetector:
    def __init__(self):
        
        self.color_ranges = {"blue": [np.array([110,50,50]), np.array([130,255,255])],
                             "green": [np.array([36, 25, 25]), np.array([70, 255,255])],
                             "red": [np.array([0, 100, 100]), np.array([10, 255, 255])],
                             "black": [np.array([0,0,0]), np.array([180, 255, 40])]
                             }                                              # Color ranges in HSV (Hue [0-180), Saturation [0-255], Value [0-255])
        self.block_contour_area_threshold = 200                             # Area threshold to detect blocks
        self.blocks_on_workbench = []                                       # Blocks on the workbench [(cx, cy, w, h, depth, color, area)]
        
        self.T_c_w, self.T_w_c = self.get_camera_homogeneous_tranforms()    # Homogeneous transform of camera frame wrt world frame and its inverse
        self.bridge = cv_bridge.CvBridge()                                  # To convert ROS image messages to OpenCV images
        self.image_height, self.image_width = self.get_image_dimensions()
        self.pin_cam = self.get_pinhole_camera_model()                      # Pinhole camera model of the kinect camera
        self.depth_image = self.get_depth_image()
        self.workbench_depth = self.get_workbench_depth()
        
        
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
        self.detected_objects_pub = rospy.Publisher('/object_detection', DetectedObjectsStamped, queue_size=10)
    
    def get_image_dimensions(self) -> Tuple[int]:
        """Computes the image height and width in pixels."""
        image = self.get_color_image()
        h, w, c = image.shape
        return h, w

    def get_camera_homogeneous_tranforms(self) -> np.ndarray:
        """Returns the homogeneous transform of the camera frame wrt world frame and its inverse.
        
        The camera frame is following the opencv convention (not the frame of the camera model in Gazebo) 
        where the Z-axis is along the optical axis.
        """
        camera_origin = self.get_model_position_from_gazebo("kinect")        # Position of origin of camera frame wrt world frame
        Rot_c_w =  np.array([[ 0, -1,  0, 0],
                             [-1,  0,  0, 0],
                             [ 0,  0, -1, 0],
                             [ 0,  0,  0, 1]])                               # Rotation of camera wrt world (Homogeneous transform)
        Transl_c_w = tf.transformations.translation_matrix((camera_origin))  # Translation of camera wrt world (Homogeneous transform)
        T_c_w =  np.dot(Transl_c_w, Rot_c_w)                                 # Homogeneous transform of camera wrt world
        T_w_c = tf.transformations.inverse_matrix(T_c_w)                     # Homogeneous transform of world wrt camera
        return T_c_w, T_w_c
        
    def get_depth_image(self) -> np.ndarray:
        """Returns current depth image in OpenCV fomrat."""
        image_msg = rospy.wait_for_message('/camera/depth/image_raw', Image, timeout=10)
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='32FC1')
        return image
    
    def get_color_image(self) -> np.ndarray:
        """Returns current color image in OpenCV fomrat."""
        image_msg = rospy.wait_for_message('/camera/color/image_raw', Image, timeout=10)
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        return image

    def get_pinhole_camera_model(self) -> PinholeCameraModel:
        """Creates a pinhole camera model from the camera's ROS parameters."""
        pin_cam = PinholeCameraModel()
        cam_info = rospy.wait_for_message('/camera/color/camera_info', CameraInfo, timeout=10)
        pin_cam.fromCameraInfo(cam_info)
        return pin_cam

    def get_model_position_from_gazebo(self, model: str) -> Tuple[float]:
        """Returns the postion of the model wrt the world frame."""
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        while not rospy.is_shutdown():
            try:
                response = model_state_client(model, "world")
                if response.success == True:
                    return response.pose.position.x, response.pose.position.y, response.pose.position.z
            except rospy.ServiceException as e:
                rospy.loginfo("Service call for GetModelState failed: {}".format(e))
            rospy.sleep(2)

    def get_mask(self, hsv: np.ndarray, color: str) -> np.ndarray:
        """Creates a mask of the image that detects the given color."""
        if color not in self.color_ranges.keys():
            raise ValueError('The requested color to mask is not on the list of detectable colors.')

        mask = cv2.inRange(hsv, self.color_ranges[color][0], self.color_ranges[color][1])  # Check if array elements lie between the bounds. Returns 255 if yes, 0 otherwise. 
        return mask

    def get_workbench_depth(self) -> float:
        """Computes the depth of the workbench."""

        # Detect black regions (the workbench color) in image
        image = self.get_color_image()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                                       # Convert image from BGR to HSV color-space
        mask = self.get_mask(hsv, "black")
        # All the detected black regions will be white in the mask. Loop through mask, to find a white pixel (value 255).
        height, width = mask.shape
        flag = False
        u, v = None, None
        for i in range(height):
            for j in range(width):
                if mask[i, j] == 255:
                    u, v = i, j
                    flag = True
                    break
            if flag:
                break

        # Get depth value of that pixel
        depth = self.get_pixel_depth(u, v)
        return depth

    def get_pixel_depth(self, u: float, v: float) -> float:
        """Returns the value of the pixel in the depth image."""
        if u >= self.image_width:
            u = self.image_width - 1
        if v >= self.image_height:
            v = self.image_height - 1

        return self.depth_image[int(v), int(u)]
        
    def compute_mass_center(self, image_array: np.ndarray) -> Tuple[int]:
        """Computes the center of mass of the image."""
        M = cv2.moments(image_array)
        cx = int(M['m10']/(M['m00'] + 1e-6))  # Add 1e-6 to avoid division by zero
        cy = int(M['m01']/(M['m00'] + 1e-6))
        return cx, cy

    def get_detected_objects(self, contour_images: List, color: str) -> List[Tuple]:
        """Returns all objects of the given color from the contour images.
        
        cx, cy:  Pixel coordinates of the center of mass
        w, h:    Width and height in pixels of the bounding box surounding the object
        height:  Height of object
        color:   Color of the object
        area:    Area of the object's contour
        """
        objects = []
        for image in contour_images:
            area = cv2.contourArea(image)
            if area > self.block_contour_area_threshold:
                cx, cy = self.compute_mass_center(image)
                _, _, w, h = cv2.boundingRect(image)
                depth = self.get_pixel_depth(cx, cy)
                height = self.workbench_depth - depth
                objects.append((cx, cy, w, h, height, color, area))
        return objects
    
    def get_3D_point_from_pixel(self, u: float, v: float) -> Tuple[float]:
        """Computes the coordinates of the 3D point in the world frame that corresponds to the given pixel.
        
        We assume the point is on an object placed on top of the workbench. 
        So, we compute the Z-axis coordinate at half of its height (center of mass).
        """

        # Get object height
        depth = self.get_pixel_depth(u, v)
        height = self.workbench_depth - depth
        
        # Get point in homogeneous coordinates wrt camera frame (up to scale)
        ray = self.pin_cam.projectPixelTo3dRay((u, v))
        
        # Convert point to cartesian coordinates wrt camera frame (up to scale) (Normalize ray so Z-component equals 1)
        X_ray = ray[0]/ray[2]
        Y_ray = ray[1]/ray[2]

        # Since we know the object's depth (therefore the point's Z coordinate wrt camera), we can compute the exact 3D point
        # Get point in cartesian coordinates wrt camera frame (x=X/Z, y=Y/Z -> X=x*Z, Y=y*Z)
        Z = depth + height/2
        X = X_ray*Z 
        Y = Y_ray*Z

        # Get point in cartesian coordinates wrt world frame
        X_world, Y_world, Z_world = self.convert_point_from_camera_to_world(X, Y, Z)

        return X_world, Y_world, Z_world
    
    def get_pixel_from_3D_point(self, x_world: float, y_world: float, z_world: float) -> Tuple[float]:
        """Computes pixel values from the 3D coordinates of a point in the world frame.

        project3dToPixel() expects the points to be wrt the camera frame defined in camera_info (here camera_link)
        """
        x, y, z = self.convert_point_from_world_to_camera(x_world, y_world, z_world)  # Point wrt camera frame
        u, v = self.pin_cam.project3dToPixel((x,y,z))
        return u, v

    def convert_point_from_camera_to_world(self, x: float, y: float, z: float) -> Tuple[float]:
        """Converts point coordinates from camera frame to world frame. 
        
        The camera frame is following the opencv convention (not the frame of the camera model in Gazebo) 
        where the Z-axis is along the optical axis.
        (x_world, y_world, z_world = x_camera_origin - y, y_camera_origin - x, z_camera_origin - z)
        """
        X_c = np.array([x, y, z, 1])                        # Point wrt camera frame
        X_world = np.dot(self.T_c_w, X_c)                   # Point wrt world frame
        return X_world[0], X_world[1], X_world[2]
    
    def convert_point_from_world_to_camera(self, x_world: float, y_world: float, z_world: float) -> Tuple[float]:
        """Converts point coordinates from world frame to camera frame. 
        
        The camera frame is following the opencv convention (not the frame of the camera model in Gazebo) 
        where the Z-axis is along the optical axis.
        (x, y, z = y_camera_origin - y_world, x_camera_origin - x_world, z_camera_origin - z_world)
        """
        X_world = np.array([x_world, y_world, z_world, 1])  # Point wrt world frame
        X_c = np.dot(self.T_w_c, X_world)                   # Point wrt camera frame
        return X_c[0], X_c[1], X_c[2]

    def image_callback(self, msg: Image) -> None:
        """Detects objects from images based on color and publishes relevant information."""

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')  # Convert image to OpenCV fomrat
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)                     # Convert image from BGR to HSV color-space.
        
        color_contours = []
        self.blocks_on_workbench = []
        for color in ["red", "green", "blue"]:
            contours = self.get_contours(hsv, color=color)
            detected_objects = self.get_detected_objects(contours, color)
            color_contours.append(contours)
            self.blocks_on_workbench += detected_objects

        # Draw bounding boxes on image
        for cx, cy, w, h, _, color, _ in self.blocks_on_workbench:
            cv2.rectangle(image, (cx-int(w/2),cy-int(h/2)), (cx+int(w/2),cy+int(h/2)), color=(255, 255, 255), thickness=1)
            cv2.putText(image, "{} ({}, {})".format(color, int(cx), int(cy)), (int(cx-45), int(cy+30)), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)

        # We could also draw the object contours (in case objects not rectangles)
        # for contours in color_contours:
        #     cv2.drawContours(image, contours, contourIdx=-1, color=(255,255,255), thickness=1)

        cv2.imshow("Camera View", image)
        cv2.waitKey(1)

        self.publish_detected_objects()
    
    def publish_detected_objects(self) -> None:
        """Publishes information about the detected objects."""

        blocks = DetectedObjectsStamped() 
        blocks.header.stamp = rospy.Time.now()
        blocks.detected_objects = []
        for cx, cy, w, h, height, color, area in self.blocks_on_workbench:
            X_world, Y_world, Z_world = self.get_3D_point_from_pixel(cx, cy)
            width, length = self.get_box_dimensions(cx, cy, w, h)

            detected_block = DetectedObject()
            detected_block.x_world = X_world
            detected_block.y_world = Y_world
            detected_block.z_world = Z_world
            detected_block.width = width
            detected_block.length = length
            detected_block.height = height
            detected_block.color = color

            blocks.detected_objects.append(detected_block)
        
        self.detected_objects_pub.publish(blocks)

    def get_box_dimensions(self, cx: float, cy: float, w: float, h: float) -> Tuple[float]:
        """Computes the cartesian width and length of the box from pixel coordinates.
        
        cx, cy:  Pixel coordinates of the center of mass
        w, h:    Width and height in pixels of the bounding box surounding the object
        """
        u1 = cx - int(w/2)
        v1 = cy - int(h/2)
        u2 = cx + int(w/2)
        v2 = cy + int(h/2)

        x1_world, y1_world, _ = self.get_3D_point_from_pixel(u1, v1)
        x2_world, y2_world, _ = self.get_3D_point_from_pixel(u2, v2)
        length = abs(x1_world - x2_world)
        width = abs(y1_world - y2_world)
        return width, length

    def get_contours(self, hsv: np.ndarray, color: str) -> List:
        """Creates contours of all objects in the given color."""
        mask = self.get_mask(hsv, color)
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


if __name__ == "__main__": 
    rospy.init_node("vision_object_detector", anonymous=False)
    VisionObjectDetector()
    rospy.spin()
