import random
from os import error
import roboticstoolbox as rtb
from spatialmath import SE3, SO3, Twist3
from math import pi
import numpy as np


# Panda parameters from https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
def forwardKinematicsPoESpace(q):
    T = SE3()
    # This is the matric we wish to return
    M = SE3(0.0880,0,0.333+0.3160+0.3840-0.1070)*SE3.Rx(pi)
    # Using the diagram, we can see the end effector is 0.088 in the x,
    # Nothing in the y, and the total height minus 0.1070 in the z,
    # This will be our M
    # Let us now set up the twists, Each twist requires arg=v, and w=w(from
    # the lectures). In each I will mention why I picked w and q, but will
    # just do the twist formula v = -q x q using a numpy cross.
    T1 = Twist3(arg=np.cross([0,0,-1],[0,0,0.333]), w=[0,0,1])
    # Joint one points in the positive Z axis[w=(0,0,1)], and is 0.0333 above
    # the base, so q = (0,0,0.333)
    T2 = Twist3(arg=np.cross([0,-1,0],[0,0,0.333]),w=[0,1,0])
    # Joint two points in the positive y axis[w=(0,1,0)], and is at the same
    # positiona s joint one so q=(0,0,0.333)
    T3 = Twist3(arg=np.cross([0,0,-1],[0,0,0.333+0.3160]), w = [0,0,1])
    # Joint three points in the positive Z axis[w=(0,0,1)], and is 0.3160, above
    # joint two, so q = (0,0,0.333+0.3160)
    T4 = Twist3(arg=np.cross([0,1,0],[0.0825,0,0.3160+0.333]),w=[0,-1,0])
    # Joint four
    # points opposite of the base y, so w=(0,-1,0), 0.825 to the right,
    # of joint three, so q = (0,0.0825,0.333+0.3160)
    T5 = Twist3(arg=np.cross([0,0,-1],[0,0,0.3160+0.333+0.3840]), w = [0,0,1])
    # Joint five points up, so w=(0,0,1) and
    # is 0.0825 to the left of joint 4, and 0.3840 above, so
    # q = (0.0825-0.0825,0,0.3160+0.333+0.3840)
    T6 = Twist3(arg=np.cross([0,1,0],[0,0,0.3160+0.333+0.3840]), w=[0,-1,0])
    # Joint six points opposite of the base y so w=(0,-1,0)
    # and is at the same place  as joint 5 so
    # q = (0.0825-0.0825,0,0.3160+0.333+0.3840)
    T7 = Twist3(arg=np.cross([0,0,1],[0.088,0,0.3840+0.3160+0.333]), w=[0,0,-1])
    # Joint six points downwards of the base z so w=(0,0,-1)
    # and is at the 0.088 right of joint 6, so
    # q = (0.088,0,0.3160+0.333+0.3840)
    # Now that each twist is found, I simply write out the exponential formula in
    # lectures, with M in the start and then use the built in matrix-multiply.
    T = T1.exp(q[0])*T2.exp(q[1])*T3.exp(q[2])*T4.exp(q[3])*T5.exp(q[4])*T6.exp(q[5])*T7.exp(q[6])*M

    return T


def forwardKinematicsPoEBody(q):
    T = SE3()
    # This is the matric we wish to return
    M = SE3(0.0880,0,0.333+0.3160+0.3840-0.1070)*SE3.Rx(pi)
    # Using the diagram, we can see the end effector is 0.088 in the x,
    # Nothing in the y, and the total height minus 0.1070 in the z,
    # This will be our M
    # Let us now set up the twists, Each twist requires arg=v, and w=w(from
    # the lectures). In each I will mention why I picked w and q, but will
    # just do the twist formula v = -q x q using a numpy cross.
    # Compared to the first one, we use the end effector as our reference
    T7 = Twist3(arg=np.cross([0,0,-1],[0,0,-0.1070]), w=[0,0,1])
    # Joint 7 points downwards in the end effactor, so w=(0,0,1) while it is
    # 0.1070 above(negative z), so q=(0,0,-0.1070)
    T6 = Twist3(arg=np.cross([0,-1,0],[-0.088,0,-0.1070]), w=[0,1,0])
    # Joint 6 points out of the page, so the y of the ef, so w=(0,1,0)
    # while it 0.088 to the left(negative x) of joint 6, so q = [-0.088,0,-0.1070]
    T5 = Twist3(arg=np.cross([0,0,1],[-0.088,0,-0.1070]), w = [0,0,-1])
    # Joint 5 is pointing upward, so w=(0,0,-1) and is at the same position as
    # joint 6 so q= [-0.088,0,-0.1070]
    T4 = Twist3(arg=np.cross([0,-1,0],[-0.0055,0,0.3840-0.1070]),w=[0,1,0])
    # Joint 4 is pointing in the positive y while is 0.3840 below  and
    # 0.0825 to the right(positive x) of joint 5 so
    # q=[-0.0055,0,0.3840-0.1070]
    T3 = Twist3(arg=np.cross([0,0,1],[-0.088,0,0.3840-0.1070]), w = [0,0,-1])
    # Joint 3 is pointing in the negative z so q=(0,0,-1)
    # while 0.0825 of the left of joint
    # 4 so q=[-0.088,0,0.3840-0.1070])
    T2 = Twist3(arg=np.cross([0,1,0],[-0.088,0,0.3160+0.3840-0.1070]),w=[0,-1,0])
    # Joint 2 points in in the page so w=(0,-1,0) and is another 0.3160 below
    # Joint 3 so q=[-0.088,0,0.3160+0.3840-0.1070]
    T1 = Twist3(arg=np.cross([0,0,1],[-0.088,0,0.3160+0.3840-0.1070]), w=[0,0,-1])
    # Joint 1 points upward so w=(0,0,-1) and is at the same position as
    # joint 2 so q=[-0.088,0,0.3160+0.3840-0.1070]

    # I simply did the same procedure in the previous function but
    # used the body formula with M at the start.
    T = M*T1.exp(q[0])*T2.exp(q[1])*T3.exp(q[2])*T4.exp(q[3])*T5.exp(q[4])*T6.exp(q[5])*T7.exp(q[6])
    return T


def randomize_param(num, min, max):
    # This function simply takes a number of slices(num)
    # A minimum, and maximum, and picks a number in that range using
    # num slices
    rand_int = random.randint(0,num)
    # We pick the number of slices to go
    slice = (max-min)/num
    # This is the distance between the two parameters
    # we then return that many above the minimum
    return (slice*rand_int+min)


def evalManufacturingTolerances():
    error_list = [[],[],[],[],[],[]]
    # This is the list storing the error value for each parameter in this order:
    # [x_pos,y_pos,z_pos,x_rot,y_rot,z_rot]
    # We will append to each list throughout.
    robot_normal = rtb.models.DH.Panda()
    robot_defec = rtb.models.DH.Panda()
    #We create the normal robot, and the robot with the defects above
    robot_defec.links[0].d = 0.383
    # We set the d value of the first link(joint) to be 0.383
    robot_defec.links[2].alpha = 1.5807
    # We set the alpha value of the 3rd link(joint) to be 1.5807
    # This makes it so the robot_defec robot now has the wrong configuration
    for i in range(1000):
        t1 = randomize_param(40, -2.8973, 2.8973)
        t2 = randomize_param(40, -1.7628, 1.7628)
        t3 = randomize_param(40, -2.8973, 2.8973)
        t4 = randomize_param(40, -3.0718, -0.06980)
        t5 = randomize_param(40, -2.8973, 2.8973)
        t6 = randomize_param(40, -0.01750, 3.7525)
        t7 = randomize_param(40, -2.8973, 2.8973)
        # For each parameter, I randomized the value out of 40 slices
        # from the maximum and minimum value of each joint. This is given
        # on the franka robotics website:
        # # Panda parameters from https://frankaemika.github.io/docs/control_parameters.html#denavithartenberg-parameters
        pos = [t1,t2,t3,t4,t5,t6,t7]
        # I setup a matrix containing all of the parameter values
        b = robot_normal.fkine(pos)
        d = robot_defec.fkine(pos)
        # I then calculated the forward kinematics of both robotcs using fkine
        b_inv = b.inv()
        Tbd = b_inv*d
        # Here, I solve for the matrix from the normal position, to the defected
        # robot(Tbd), since we wish to find the error from the normal, I did the following:
        #  T_{bd} = T_{bs} T_{sd} = T_{sb)^{-1} T_{sd}
        # To find the matrix, we just multiply the inverse of the normal
        # with the defect, which is done and stored in matrix tbd.
        inv_twist = Tbd.twist()
        # We then find the twist to move the end effector from b to d, by
        # finding the twist of #tbd
        for i in range(6):
            # We then loop through all six values in the twist, and
            # append them to the according list inside of the error list
            error_list[i].append(inv_twist.A[i])
    mean_standard_max_list = []
    # We will now take the data for each of the 6 parameters,
    # Find the mean, std_div, and max of each, put it in a list and
    # append it here
    for i in range(6):
        # For each data, we set the current list
        curr_list = error_list[i]
        # We find the mean
        mean = np.mean(curr_list)
        # We find the standard deviation
        std_div = np.std(curr_list)
        # We find the max
        max_in_list = max(curr_list)
        # We put all three in a list and put it to the info list
        mean_standard_max_list.append([mean,std_div, max_in_list])
    # We now have a list containing lists that have the mean, std, div, and
    # max of each parameter in the twist, let us now print them.
    print("\t Mean \t \t Standard_Div \t \t Max")
    values = ["x_pos", "y_pos", "z_pos", "x_rot","y_rot","z_rot"]
    for i in range(6):
        curr_list = mean_standard_max_list[i]
        print(values[i] + "\t" +str(curr_list[0]) + "\t" + str(curr_list[1]) + "\t" + str(curr_list[2]))

# Load the robot model
robot = rtb.models.DH.Panda()


t1 = randomize_param(40, -2.8973, 2.8973)
t2 = randomize_param(40, -1.7628, 1.7628)
t3 = randomize_param(40, -2.8973, 2.8973)
t4 = randomize_param(40, -3.0718, -0.06980)
t5 = randomize_param(40, -2.8973, 2.8973)
t6 = randomize_param(40, -0.01750, 3.7525)
t7 = randomize_param(40, -2.8973, 2.8973)
# robot.qz = [t1,t2,t3,t4,t5,t6,t7]
# This just sets up random params to test fkine, I have turned it off
# to leave it close to the source code.

robot.tool = SE3() # we remove the tool offset as we are only interesed in the FK to the end-effector


# Q1.1
poe = forwardKinematicsPoESpace(robot.qz)
print("\n FK POE space", poe)
# poe = forwardKinematicsPoESpace2(robot.qz)
# print("\n FK POE space", poe)

# Q1.2
poe_b = forwardKinematicsPoEBody(robot.qz)
print("\n FK POE body", poe_b)

# You can compare your implementation of the forward kinematics with the one from prtb
# test with multiple valid robot configurations
dhres = robot.fkine(robot.qz)
print("\n FK DH",dhres)





# Q1.3
evalManufacturingTolerances()
