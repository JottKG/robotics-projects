import roboticstoolbox as rtb
import swift
import numpy as np
import time
from spatialmath import SE3, SO3, Twist3
from math import pi
import spatialgeometry as sg

# Robot 1 is 0.6 in the x of the origin, and the point of reference is
# rotated by pi
robot_1_pos_se3 = SE3(0.6,0,0)*SE3.Rz(pi)
# Robot 2 is similarly 0.5 behind in x, and 0.1 in y of robot 2
robot_2_pos_se3 = SE3(-0.5,-0.1,0)
# We then make robots 1 and 2
robot1 = rtb.models.Panda()
robot2 = rtb.models.Panda()

# We set the base of robot 1 as the position matrix above
robot1.base = robot_1_pos_se3
# Similarly, we set the base of robot 2 as the position matrix above
robot2.base = robot_2_pos_se3
# Init joint to the 'ready' joint angles for both robots
robot1.q = robot1.qr
robot2.q = robot2.qr
# Let's now make a cylinder of 0.3 length and 0.1 radius as specified
cylinder = sg.Cylinder(length = 0.3, radius = 0.01, color="red")

# Testing code, [IGNORE]
# cylinder2 = sg.Cylinder(length = 0.3, radius = 0.01, color="red")

# Launch the simulator Swift
env = swift.Swift()
env.launch()

# add the robot1 to the environment
env.add(robot1)
# add the robot2 to the environment
env.add(robot2)
# add the cylinder to the environment
env.add(cylinder)
# env.add(cylinder2)
# set the cylinder's position to the end-effector of the robot1
cylinder.T = robot1.fkine(robot1.q)


# cylinder2.T = robot1.fkine(robot1.q)

# This is our callback funciton from the sliders in Swift which set
# the joint angles of our robot1 to the value of the sliders

# CODE BELOW IS BASED ON TUTORIAL 2
def set_joint(j, value):
    # We save the joint value before updating, in case this update
    # moves out of range of panda12[as given in the note]
    prev_joint_val = robot1.q[j]
    # Update joint value of robot1 via the input value
    robot1.q[j] = np.deg2rad(float(value))
    # Here, we update the position of the cylinder, we first
    # Find the position of the end effector, rotate on Y by pi/2
    # (such that it is being held parallel to the end effector)
    # Then translate by 0.15 on the z such that the rod is sticking
    # away from the end-effector
    cylinder.T = robot1.fkine(robot1.q)*SE3.Ry(pi/2)*SE3(0,0,0.15)
    # cylinder2.T = robot1.fkine(robot1.q)*SE3.Ry(pi/2)*SE3.Rz(pi)*SE3(0,0,0.5)
    # Similarly, the end of the rod would be in the same direction as the
    # above cylinder position, except since the rod is of length 0.3, it would
    # be 0.3 in z from the end effector, we also rotate BACK in -pi/2
    # such that the end-effector of robot2 is pointing inward to the end
    # of the rod instead of inside it.
    pos_end = robot1.fkine(robot1.q)*SE3.Ry(pi/2)*SE3(0,0,0.3)*SE3.Ry(-pi/2)
    # We then want to convert the current position of the coordinate and
    # rotation in reference to robot 1 to reference in robot 2
    # We simply inverse the matrix to find robot 2 to origin
    # Then use the position from origin to the end-effector to find
    # The robot to end-effector position
    pos = robot_2_pos_se3.inv()*pos_end
    # We then do the inverse kinematics on this position in robot2,
    # Nothing Fancy. For the jittering, I used this specific vector because it
    # is basically the default vector which is kind of guaranteed to be
    # somewhat much nearer to whatever position we have then a normal
    # guess. I tested this with all joint vlaues and it did not jitter.
    # I know there is a possibility of adding or changing the vector
    # input depending on other parameters but one seemed to work well.
    inv_kin = robot2.ikine_LM(pos,q0=[1.66320,-0.39798,-0.81156,-2.03811,-0.19019,1.712247,-1.14774])
    # If the inv kine has not failed, we will update the position of robot2
    # When I was testing, failure sometimes took a little while tom
    # compute which added a delay.
    if inv_kin.success==True:
        robot2.q = inv_kin.q
    else:
        # If this fails, then reset this joint value back to being okay.
        robot1.q[j] = prev_joint_val
        # Undo the transformation for the cylinder
        cylinder.T = robot1.fkine(robot1.q)*SE3.Ry(pi/2)*SE3(0,0,0.15)
        # I think this is what was meant by the note, piazza post @87
        # says that the move can be if the slider is let go.
        # The handout just says "adjust" the values, which I assume means
        # you can do this in a myriad of ways, while the most obvious
        # one I had was just denying the movement. I could have
        # also just found the closest suitable position[ but this would
        # have been very resource intensive on my tests because of the amount
        # of position values we would have to go through in ikine ]


# Loop through each link in the robot1 and if it is a variable joint,
# add a slider to Swift to control it
j = 0
for link in robot1.links:
    if link.isjoint:
        # We use a lambda as the callback function from Swift
        # j=j is used to set the value of j rather than the variable j
        # We use the HTML unicode format for the degree sign in the unit arg
        env.add(
            swift.Slider(
                lambda x, j=j: set_joint(j, x),
                min=np.round(np.rad2deg(link.qlim[0]), 2),
                max=np.round(np.rad2deg(link.qlim[1]), 2),
                step=1,
                value=np.round(np.rad2deg(robot1.q[j]), 2),
                desc="robot1 Joint " + str(j),
                unit="&#176;",
            )
        )


        j += 1


while True:
    # Update the environment with the new robot1 pose
    env.step(0)
    time.sleep(0.01)
