#!/usr/bin/env python

"""
aptag_optimized.py

Determines the position and orientation of the robot within the arena. There are four apriltags
pasted on the four vertical sides of the robot, which a camera on the server observes. This node
reads from that camera's stream and processes it using OpenCV to determine the aforementioned
parameters. 

For a more in-depth description of localization, reference the README.

TODO:
- Debug issue where orientation jumps around
- Figure out which way x and y are relative to the arena walls
"""

from __future__ import division
from __future__ import print_function
from rdt_localization.msg import Pose
import rospy
import cv2
import apriltag
import numpy
import math
import time
import turtle
import copy

'''
Values come from running camera calibration file (fx, fy, cx,cy).
Camera should be calibrated for each new computer and new camera.
If the projected green box for the apriltag does not have sudden jumps when you turn the tag slightly,
then the calibration is correct.
Sometimes, if the outputs aren't correct, try calibrating with a higher number of detections. 
The better the calibration, the less you need to adjust the scale of the measurements in center_position()
'''
# camera_params = [1.01446618 * 10 ** 3, 1.02086461 * 10 ** 3, 6.09583146 * 10 ** 2, 3.66171174 * 10 ** 2]
camera_params = [825.90832362, 823.90765969, 418.65176814, 206.27259118]

# All distance and lengths in meters
# All angles output converted to degrees

'''
Converts rotation matrix into radians and degrees
id is id of apriltag
If R is rotation matrix:
R = Rx times Ry times Rz
ex. Rx = [1, 0, 0]
         [0, cos(xangle), sin(xangle)]
        [0, sin(xangle), cos(xangle)]
Function taken from: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
'''
def rotation_matrix_to_euler_angles(R, tag_id):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])

    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # Set which ids are on which side

    # Back
    if tag_id == 3:
        return numpy.array([math.degrees(x), math.degrees(y), math.degrees(z)])

    # Right
    elif tag_id == 4 or tag_id == 5:
        return numpy.array([math.degrees(x), math.degrees(y) + 90, math.degrees(z)])

    # Front
    elif tag_id == 0:
        return numpy.array([math.degrees(x), math.degrees(y) + 180, math.degrees(z)])

    # Left
    elif tag_id == 1 or tag_id == 2:
        return numpy.array([math.degrees(x), math.degrees(y) + 270, math.degrees(z)])

    # Call function without predefined rotations
    elif tag_id == -1:
        return y

    elif tag_id == 6:
        return numpy.array([math.degrees(x), math.degrees(y), math.degrees(z)])
        


'''
Angular rotation of each apriltag
position = position matrix
Output: angle in degrees centered at the center of the camera. Left bound angles are negative (0, -90) degrees
Right bound angles are positive (0, 90) degrees
'''
def find_center_angular_rotation(position):
    angle = math.atan(position[0] / position[1])
    return math.degrees(angle)


'''
Finds center position of robot using position matrix, rotation matrix, and tag id
diagonal_length is distance from center of robot to center of apriltag in meters
default_angle is angle formed between diagonal_length and side of robot
Requires calibration to ensure output is correctly scaled--> meter_length_adjustment is a percentage multiplier
that adjusts the meters calculated by the robot. At 10 meters: measurement10m - 10m / 10 = how much each 
meter is offset by. meter_length_adjustment = 1 - offset
Works for test robot, which is a rectangle with 2 apriltags on each of the long sides and one apriltag 
for each of the short sides. Returns center position (x,y), which is the position of the robot from a bird's view
perspective. x is the horizontal position from the center of the camera and y is the vertical position

Variables: diagonal_length, default_angle, meter_length_adjustment
diagonal_length is the distance from the center of the apriltag to the center of the robot
default_angle is the angle the apriltag makes with the diagonal line
meter_length_adjustment is the scaling factor. This can be set for each individual tag depending on how precisely you 
want to calibrate the measurements
'''
def center_position(position, rotation_matrix, id):

    relative_orientation = rotation_matrix_to_euler_angles(rotation_matrix, -1)
    diagonal_length = 0.33234
    default_angle = math.radians(45)
    meter_length_adjustment = 1.01

    # Right Tag
    if id == 5 or id == 2:
        total_angle = math.pi - default_angle - relative_orientation

        if math.degrees(total_angle) > 90:
            x_position = position[0] + diagonal_length * math.cos(total_angle)
            y_position = position[2] + diagonal_length * math.sin(total_angle)
        else:
            x_position = position[0] - diagonal_length * math.cos(total_angle)
            y_position = position[2] + diagonal_length * math.sin(total_angle)

        return [numpy.float(x_position * meter_length_adjustment), numpy.float(y_position * meter_length_adjustment)]

    # Left Tag
    if id == 4 or id == 1:
        total_angle = default_angle + relative_orientation * -1

    # Only one Tag
    elif id == 0 or id == 3 or id == 6:
        meter_length_adjustment = 1.02
        diagonal_length = .50
        total_angle = -1 * relative_orientation + math.radians(90)
        print(diagonal_length * math.cos(total_angle))
        print(math.degrees(total_angle))

    if math.degrees(total_angle) > 90:
        x_position = position[0] - diagonal_length * math.cos(total_angle)
        y_position = position[2] + diagonal_length * math.sin(total_angle)

    else:
        x_position = position[0] + diagonal_length * math.cos(total_angle)
        y_position = position[2] + diagonal_length * math.sin(total_angle)

    return [numpy.float(x_position * meter_length_adjustment), numpy.float(y_position * meter_length_adjustment)]


'''
Takes the stored_pose list and finds the index of the smallest angle in the list.
Creates a list of rotation matrices, then makes a list of the angles. These will keep the original index order since
the lists are appended. A deep copied list and a sorted list are made for the orientation_list. The first value 
(smallest value) is compared to the copied (original) list and the index is found.
'''
def find_smallest_rotation_index(stored_pose):
    rotation_matrix_list = []

    for stored_index in range(len(stored_pose)):
        rotation_matrix = numpy.array([stored_pose[stored_index][0][:3],
                                       stored_pose[stored_index][1][:3],
                                       stored_pose[stored_index][2][:3]])
        rotation_matrix_list.append(rotation_matrix)

    orientation_list = []
    for matrix in rotation_matrix_list:
        angle = math.fabs(rotation_matrix_to_euler_angles(matrix, -1))
        orientation_list.append(angle)

    original_orientation_list = copy.deepcopy(orientation_list)

    orientation_list.sort()
    smallest_angle_index = original_orientation_list.index(orientation_list[0])

    return smallest_angle_index


# Prints all data determined by the program
def debug_log(detection, pose, init_error, final_error, num_detections, i):
    # Converts rotation matrix of individual apriltag into degrees and radians
    print('Detection {} of {}:'.format(i + 1, num_detections))
    print()
    print(detection.tostring(indent=2))
    print()
    print('pose:', pose)
    print('init error, final error:', init_error, final_error)
    print()


# Performs/calls for all calculations
def calc_values(stored_pose, stored_id):

    if len(stored_pose) > 1:
        smallest_rotation_index = find_smallest_rotation_index(stored_pose)

    else:
        smallest_rotation_index = 0

    rotation_matrix = numpy.array([stored_pose[smallest_rotation_index][0][:3],
                                   stored_pose[smallest_rotation_index][1][:3],
                                   stored_pose[smallest_rotation_index][2][:3]])

    position = numpy.array([stored_pose[smallest_rotation_index][0][3:],
                            stored_pose[smallest_rotation_index][1][3:],
                            stored_pose[smallest_rotation_index][2][3:]])

    orientation = rotation_matrix_to_euler_angles(rotation_matrix, stored_id[smallest_rotation_index])[1]

    center_coords = center_position(position, rotation_matrix, stored_id[smallest_rotation_index])

    angular_rotation = find_center_angular_rotation(center_coords)

    return rotation_matrix, orientation, center_coords, angular_rotation


# Maps coordinates and rotation of robot to turtle.
# scale adjusted to set distance over travelled pixel ratio
def turtle_draw(coordinates, orientation, angular_rotation):
    scale = 100
    turtle.clear()
    turtle.setheading(orientation * -1 + 90)
    turtle.goto(round(coordinates[0], 2) * scale, round(coordinates[1], 2) * scale)

    turtle.write(['CP', [round(coordinates[0], 2), round(coordinates[1], 2)], 'O:',
                  round(orientation, 2), 'AR', round(angular_rotation, 2)],
                 False, align='center', font=('Arial', 25, 'normal'))


def main():
    global camera_params

    # Setup ROS Node
    pub = rospy.Publisher('server/localization', Pose, queue_size=10)
    rospy.init_node('aptag_optimized')
    rate = rospy.Rate(10)

    # Setup turtle
    # turtle.clear()
    # turtle.setworldcoordinates(-500, 0, 500, 1000)
    # turtle.penup()
    # turtle.speed(5)
    # turtle.turtlesize(2, 2, 1)

    # Set value for camera
    cap = cv2.VideoCapture(0)

    # Name for window
    window = 'Camera Detection Window'
    cv2.namedWindow(window)

    detector = apriltag.Detector()

    # Apriltag detection loop
    while not rospy.is_shutdown():
        # Stores values of each detected tag for final output
        stored_pose = []
        stored_id = []

        success, frame = cap.read(1)
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        detections, dimg = detector.detect(gray, return_image=True)
        print()
        num_detections = len(detections)
        print('Detected {} tags.\n'.format(num_detections))

        # Detection for each apriltag
        for i, detection in enumerate(detections):

            # Returns pose matrix
            # Set tagsize in meters here
            pose, init_error, final_error = detector.detection_pose(detection, camera_params, tag_size=0.17, z_sign=1)

            # debug_log(detection, pose, init_error, final_error, num_detections, i)

            if num_detections > 0:
                stored_pose.append(pose)
                stored_id.append(detection.tag_id)

            overlay = frame // 2 + dimg[:, :, None] // 2

            # Draws all overlays before going to next frame. Must import cv2 in apriltag module
            #_draw_pose(overlay, camera_params, tag_size, pose, z_sign=1)
            #for det in range(num_detections):
            	#apriltag._draw_pose(overlay, camera_params, 0.17, pose, z_sign=1)

            cv2.imshow(window, overlay)
            cv2.waitKey(27)

        # Outputs only if tags are detected
        if num_detections > 0:
            #time.sleep(1)
            rotation_matrix, orientation, center_coords, angular_rotation = calc_values(stored_pose, stored_id)

            # print('Center Position:\n', center_coords)
            # print()
            # print('Orientation\n', orientation)
            # print()
            # print('Angular Rotation:\n', angular_rotation)

            outmsg = Pose()
            outmsg.x = center_coords[0]
            outmsg.y = center_coords[1]
            #outmsg.x = 13.74
            #outmsg.y = -5
            outmsg.orientation = orientation

            pub.publish(outmsg)

            # turtle_draw(center_coords, orientation, angular_rotation)

            rate.sleep()


if __name__ == '__main__':
    main()
