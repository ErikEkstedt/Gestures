# XML-files defining environments

## Custom_reacher

Simple implementation of a reach with two joints. The first may only rotate in the z-direction and the second one in the y-direction.

Both arm parts are 0.2 long and the target is defined in 3d space around the arm.

Robot Joints (2 DoF):
1. Stationary joint.
	* Located (0,0,0.2)
	* No limits
	* Z-direction
2. Second joint.
	* No limits
	* Y-direction

Target Joints:
1. Stationary joint.
	* limits:
		* Z: [0.2 0.6]
	* XYZ-direction

## Custom_reacher2

Making the reacher a bit more complicated adding movement joints in all directions at each joint. 6 DoF arm. The target is the same as in the original custom_reacher.

Robot Joints (6 DoF):
1. Stationary joint.
	* No limits
	* XYZ-direction
2. Second joint.
	* No limits
	* XYZ-direction
	
Target Joints (same as above):
1. Stationary joint.
	* limits:
		* Z: [0.2 0.6]
	* XYZ-direction

## Custom_reacher3

The robot has the same complexity as the first one but now the target gets more complicated. 

Instead of defining one point in 3D space the target now consists of two points, one for each joint of the robot. The target will be a copy of the Robot in the same position (see through and no collision dymanics) as a goal for the robot arm to be in.

Robot Joints (2 DoF):
1. Stationary joint.
	* Located (0,0,0.2)
	* No limits
	* Z-direction
2. Second joint.
	* No limits
	* Y-direction

Target Joints (2 DoF):
1. Stationary joint.
	* Located (0,0,0.2)
	* No limits
	* Z-direction
2. Second joint.
	* No limits
	* Y-direction

