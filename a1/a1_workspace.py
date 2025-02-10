import numpy as np
import roboticstoolbox as rtb
from spatialmath import *
from math import pi
import math
import matplotlib.pyplot as plt
from matplotlib import cm
np.set_printoptions(linewidth=100, formatter={'float': lambda x: f"{x:8.4g}" if abs(x) > 1e-10 else f"{0:8.4g}"})

# f#Creating a DH robot
robot = rtb.DHRobot(
    [
        #Using DH joins, MDH was not working..
        # First joint is revolute, with the given parameter(theta)
        # being ommited due to being supplied in
        # Qlim uses the math.pi to set limits.
        rtb.RevoluteDH(alpha=0,a=1,d=0,qlim=[0,math.pi/2]),
        #Second one is peismatic, with the parameter(d) being
        #ommited
        rtb.PrismaticDH(alpha=0,a=0,theta=math.pi/2,qlim=[0.5,1.5]),
        #Very similar to the first one except the values are different
        rtb.RevoluteDH(alpha=0,a=1,d=0,qlim=[-math.pi/2,math.pi/2]),
    ], name="myRobot")
# q is six dimensional, and we supply THREE
#We will have theree lists for all the x,y,z values of all the positions of
#the end effector, so for the 0th position, it's x is at x_list[0],
# y is at y_list[0] and z_list[0], and so on for all n positions we will sample.
x_list=[]
y_list=[]
z_list=[]
#This is just the default position
q2 = [0, 0.5, 0]
# This is how many samples we want to take from each parameter
# I will take 8, since anything more doesn't add value, while less doesn't
# These will slice the paramters to sample 8 values of each
slices = 8

for i in range(slices+1):
    # We will loop the first parameter from "slice" 0,1,2...8
    current_SE3 = robot.fkine(q2)
    # Let us set the position to the default to prepare it.
    for j in range(slices+1):
        # We will loop the second parameter from "slice" 0,1,2...8
        current_SE3 = robot.fkine(q2)
        # Let us set it to default again
        for k in range(slices+1):
            # We will loop the third parameter from "slice" 0,1,2...8
            # We now store the x,y,z of the current position.
            xyz = [(math.pi/2)*(1/slices)*i,0.5 + (1/slices)*j,
                   -(math.pi/2) + (math.pi/slices)*k]
            # We will essentially take the first parameter from 0 to pi/2 via i
            # We will essentially take the first parameter from 0.5 to 1.5 via j
            # We will essentially take the third parameter from -pi/2 to pi/2 via k
            # All of these break up the paramaters into "slices" parts, then
            # multiplies to slowly build each position
            q2 = [0, 0.5, 0]
            current_SE3 = robot.fkine(q2)
            # Let us set the default kinematic value
            q2 = [xyz[0],xyz[1],xyz[2]]
            # Let us now use the positions above as the kinematic
            current_SE3 = robot.fkine(q2)
            # We set the robot and record the kinematics at this point
            cur_x = current_SE3.A[0][3]
            cur_y = current_SE3.A[1][3]
            cur_z = current_SE3.A[2][3]
            # We obtain the current x,y z from the SE3 matrix using
            # The notation in class
            x_list.append(cur_x)
            y_list.append(cur_y)
            z_list.append(cur_z)
            # We append the x,y,z to the lists

x_min = min(x_list)
x_max = max(x_list)
y_min = min(y_list)
y_max = max(y_list)
z_min = min(z_list)
z_max = max(z_list)
# We obtain the max of each coordinate to set up a bounding box




q1=[1, 0.5, 0]
fig = robot.plot(q1)
# We plot the default position if not done previous.

fig.ax.plot([x_min, x_max], [y_min, y_min], [z_min, z_min], color='b')
fig.ax.plot([x_max, x_max], [y_min, y_max], [z_min, z_min], color='b')
fig.ax.plot([x_max, x_min], [y_max, y_max], [z_min, z_min], color='b')
fig.ax.plot([x_min, x_min], [y_max, y_min], [z_min, z_min], color='b')

fig.ax.plot([x_min, x_max], [y_min, y_min], [z_max, z_max], color='b')
fig.ax.plot([x_max, x_max], [y_min, y_max], [z_max, z_max], color='b')
fig.ax.plot([x_max, x_min], [y_max, y_max], [z_max, z_max], color='b')
fig.ax.plot([x_min, x_min], [y_max, y_min], [z_max, z_max], color='b')

fig.ax.plot([x_min, x_min], [y_min, y_min], [z_min, z_max], color='b')
fig.ax.plot([x_max, x_max], [y_max, y_max], [z_min, z_max], color='b')
fig.ax.plot([x_min, x_min], [y_max, y_max], [z_min, z_max], color='b')
fig.ax.plot([x_max, x_max], [y_min, y_min], [z_min, z_max], color='b')
# This draws the bounding box by drawing lines on the grid, since there
# are 8 corners, we use 12 lines above using the matplotlib values
# We want to build one from the max and mins of all, so this essentially
# does that using the way you would draw it in math.
fig.ax.scatter3D(x_list, y_list, z_list, cmap='cividis')
# We now scatter all the points previously collected from all the joint
# robot positions
plt.show()
#We show the plot
fig.hold()
# We hold the plot so it does not CLOSE!
