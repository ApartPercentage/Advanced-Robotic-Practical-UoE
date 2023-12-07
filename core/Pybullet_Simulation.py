from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

# TODO: Rename class name after copying this file
class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

    prevErrors = {}
    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND'      : np.array([0, 0, 0]),
        'LHAND'      : np.array([0, 0, 0])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 0.267]),
        'HEAD_JOINT0': np.array([0, 0, 0.302]),
        'HEAD_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]),
        'LARM_JOINT1': np.array([0, 0, 0.066]),
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335]),
        'RHAND'      : np.array([0.0525, -0.029, -0.02]), # optional
        'LHAND'      : np.array([0.0525, 0.029, -0.02]) # optional
    }

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None:
           raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!")
        # TODO modify from here
        a = self.jointRotationAxis
        if jointName not in a:
        	raise Exception("Joint doesn't exist")
        i = a[jointName]
        if i[0] == 1:
            R = np.matrix([[1, 0, 0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        elif i[1] == 1:
            R = np.matrix([[np.cos(theta), 0, np.sin(theta)],[0, 1, 0],[-np.sin(theta), 0, np.cos(theta)]])
        elif i[2] == 1:
            R = np.matrix([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0, 0, 1]])
        else:
            R = np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        return R
	
    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        jointNames = list(self.frameTranslationFromParent)
        del jointNames[0:2]
        del jointNames[-2:]
        for joint in jointNames:
            theta = super().getJointPos(joint)
            R = self.getJointRotationalMatrix(joint, theta)
            p = self.frameTranslationFromParent[joint]
            rp = np.column_stack((R,p))
            transformationMatrices[joint] = np.vstack((rp, np.array([0,0,0,1])))
        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of a given joint using Forward Kinematics
            according to the topology of the Nextage robot.
        """
        
        transMatrices = self.getTransformationMatrices()
        keys = list(transMatrices)
        
        jointIndex = keys.index(jointName)
        jointList = keys[:jointIndex + 1]
        
        if 'RARM_JOINT' in jointName or 'RHAND' in jointName:
            del jointList[1:9]
        if 'LARM_JOINT' in jointName or 'LHAND' in jointName:
            del jointList[1:3]

        TM = {}
        for i in range(len(jointList)):
            if i == 0:
                TM[jointList[i]] = transMatrices[jointList[i]]
            else:
                TM[jointList[i]] = TM[jointList[i-1]]*transMatrices[jointList[i]]
                
        jointTM = TM[jointName]
        pos = np.array([jointTM[0,3], jointTM[1,3], jointTM[2,3]])
        rotmat = jointTM[0:3, 0:3]
        return pos, rotmat

    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]

    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getEFLocationAndOrientation(self, endEffector):
        transMatrices = self.getTransformationMatrices()
        jointList = list(transMatrices)
        
        if endEffector == 'RHAND':
            endEffectorPos = self.frameTranslationFromParent['RHAND']
            del jointList[1:9]
        elif endEffector == 'LHAND':
            endEffectorPos = self.frameTranslationFromParent['LHAND']
            del jointList[9:]
            del jointList[1:3]

        TM = {}
        for i in range(len(jointList)):
            jointName = jointList[i]
            if i == 0:
                TM[jointName] = transMatrices[jointName]
            else:
                jointName = jointList[i]
                TM[jointName] = TM[jointList[i-1]]*transMatrices[jointName]

        
        endEffectorTM = np.matrix([[1, 0, 0, endEffectorPos[0]],
                                   [0, 1, 0, endEffectorPos[1]],
                                   [0, 0, 1, endEffectorPos[2]],
                                   [0, 0, 0, 1]])
        baseToEFTM = TM[jointName] * endEffectorTM
        pos = np.array([baseToEFTM[0,3], baseToEFTM[1,3], baseToEFTM[2,3]])
        rotmat = baseToEFTM[0:3, 0:3]
        return pos, rotmat

    def getEFOrientation(self, endEffector):
        rotmat = self.getEFLocationAndOrientation(endEffector)[1]
        sy = math.sqrt(rotmat[0,0]*rotmat[0,0] + rotmat[1,0]*rotmat[1,0])
        thetax = math.atan2(rotmat[2,1], rotmat[2,2])
        thetay = math.atan2(-rotmat[2,0], sy)
        thetaz = math.atan2(rotmat[1,0], rotmat[0,0])
        return np.array([thetax, thetay, thetaz])
    
    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()
    
    def jacobianMatrix(self, endEffector, noChest=False):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        
        relevantJoints = list(self.jointRotationAxis)
        if endEffector == 'LHAND':
            aeff = self.jointRotationAxis['LARM_JOINT5']
            del relevantJoints[11:17]
            del relevantJoints[11]
            del relevantJoints[3:5]
        elif endEffector == 'RHAND':
            aeff = self.jointRotationAxis['RARM_JOINT5']
            del relevantJoints[3:11]
            del relevantJoints[10]
        else:
            raise Exception("Neither hands used as end effector!")

        if noChest == True:
            del relevantJoints[2]

        Jacobian_matrix_position = np.empty((0, 3), float)
        Jacobian_matrix_orientation = np.empty((0, 3), float)
        
        endEffPos = self.getEFLocationAndOrientation(endEffector)[0]
        actualJoints = list(self.jointRotationAxis)

        del actualJoints[0:2]
        del actualJoints[-2:]
        for joint in actualJoints:
            # irrelevant joints have value 0 in Jacobian
            ai = self.jointRotationAxis[joint] if joint in relevantJoints else np.array([0, 0, 0])
            jointPos = self.getJointLocationAndOrientation(joint)[0]
            displacement = endEffPos - jointPos

            crossProduct_position = np.cross(ai, displacement)
            crossProduct_orientation = ai
            
            Jacobian_matrix_position = np.append(Jacobian_matrix_position, np.array([crossProduct_position]), axis=0)
            Jacobian_matrix_orientation = np.append(Jacobian_matrix_orientation, np.array([crossProduct_orientation]), axis=0)
        
        Jacobian_matrix = np.hstack([Jacobian_matrix_position, Jacobian_matrix_orientation])
        
        Jacobian_matrix = Jacobian_matrix.T
        return Jacobian_matrix

    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, noChest=False):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            threshold: accuracy threshold
        Return: \\
            Vector of x_refs
        """
        endEffectorPosition = self.getEFLocationAndOrientation(endEffector)[0]
        dy = targetPosition - endEffectorPosition
        
        if endEffector == 'LHAND':
            EFOrientationMatrix = self.getEFLocationAndOrientation('LHAND')[1]
        else:
            EFOrientationMatrix = self.getEFLocationAndOrientation('RHAND')[1]

        thetax = orientation[0]
        thetay = orientation[1]
        thetaz = orientation[2]
        Rx = np.matrix([[1, 0, 0],
                        [0, np.cos(thetax), -np.sin(thetax)],
                        [0, np.sin(thetax), np.cos(thetax)]])
        Ry = np.matrix([[np.cos(thetay), 0, np.sin(thetay)],
                        [0, 1, 0],
                        [-np.sin(thetay), 0, np.cos(thetay)]])
        Rz = np.matrix([[np.cos(thetaz), -np.sin(thetaz), 0],
                        [np.sin(thetaz), np.cos(thetaz), 0],
                        [0, 0, 1]])
        desiredOrientationMatrix = Rz*Ry*Rx
        m = desiredOrientationMatrix * EFOrientationMatrix.T
        param = (m[0,0]+m[1,1]+m[2,2]-1)/2
        if param > 1:
            param = 1
        nue = math.acos(param)
        if nue == 0:
            do = np.array([0, 0, 0])
        else:
            r = 1/(2*math.sin(nue)) * np.array([m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]])
            do = r * math.sin(nue)
        
        J = self.jacobianMatrix(endEffector, noChest)
        dy_do = np.concatenate((dy, do))
        dTheta = np.linalg.pinv(J) @ dy_do
        return dTheta

    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        jointNames = list(self.frameTranslationFromParent)
        del jointNames[0:2]
        del jointNames[-2:]
        current_q = np.zeros(len(jointNames))
        for i in range(len(jointNames)):
            current_q[i] = super().getJointPos(jointNames[i])
        endEffectorPosition = self.getEFLocationAndOrientation(endEffector)[0]
        dist = np.linalg.norm(targetPosition - endEffectorPosition)
        num = dist * (self.updateFrequency / speed)
        interpolationSteps = round(min(maxIter, num))
        steps = np.linspace(endEffectorPosition, targetPosition, interpolationSteps)
        currentOrientation = self.getEFOrientation(endEffector)
        oriSteps = np.linspace(currentOrientation, orientation, interpolationSteps)
        pltTime = [0]
        pltDistance = [dist]
        startingTime = time.time()
        for i in range(1, interpolationSteps):
            prev_q = np.copy(current_q)
            dTheta = self.inverseKinematics(endEffector, steps[i], oriSteps[i], False)
            current_q += dTheta
            for j in range(len(jointNames)):
                self.jointTargetPos[jointNames[j]] = current_q[j]
            self.tick_without_PD()
            endEffectorPosition = self.getEFLocationAndOrientation(endEffector)[0]
            currentTime = time.time() - startingTime
            pltTime.append(currentTime)
            distanceToTarget = np.linalg.norm(targetPosition - endEffectorPosition)
            pltDistance.append(distanceToTarget)
            if distanceToTarget < threshold:
                break
        return pltTime, pltDistance

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        jointNames = list(self.frameTranslationFromParent)
        del jointNames[0:2]
        del jointNames[-2:]
        for joint in jointNames:
            self.p.resetJointState(self.robot, self.jointIds[joint], self.jointTargetPos[joint])
        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, prev_error, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        error = x_ref - x_real
        ut = (kp * error) + (kd * (error - prev_error)/self.dt) + (ki * integral)
        return ut

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral, prev_error):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            torque = self.calculateTorque(x_ref, x_real, prev_error, integral, kp, ki, kd)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            if joint == 'LARM_JOINT1' or joint == 'RARM_JOINT1':
                pass
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            pltTarget.append(x_ref)
            pltTorque.append(torque)
            pltPosition.append(x_real)
            pltVelocity.append(dx_real)
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)
            return x_ref - x_real

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = np.arange(1000)*self.dt, [], [], np.arange(1000)*self.dt, [], []
        currentVelocity = 0
        prevError = targetPosition - super().getJointPos(joint)
        for i in range(1000):
            prevPos = super().getJointPos(joint)
            prevError = toy_tick(targetPosition, super().getJointPos(joint), targetVelocity, currentVelocity, 0, prevError)
            currentVelocity = (super().getJointPos(joint) - prevPos)/self.dt

        print(prevError)
        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector1, targetPosition1, endEffector2=None, targetPosition2=None, speed=0.01, orientation1=None, orientation2=None, threshold=1e-3, maxIter=3000, debug=False, verbose=False, trajectory1=None, trajectory2=None, noChest=False, incrementalOrientation=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        jointNames = list(self.frameTranslationFromParent)
        del jointNames[0:2]
        del jointNames[-2:]
        current_q = np.zeros(len(jointNames))
        for i in range(len(jointNames)):
            current_q[i] = super().getJointPos(jointNames[i])
        endEffector1Position = self.getEFLocationAndOrientation(endEffector1)[0]
        dist = np.linalg.norm(targetPosition1 - endEffector1Position)
        num = dist * (self.updateFrequency / speed)
        interpolationSteps = round(min(maxIter, num))
        steps1 = np.linspace(endEffector1Position, targetPosition1, interpolationSteps)
        if trajectory1.size != 0:
            traj1 = trajectory1
        else:
            traj1 = steps1
        
        currentOrientation1 = self.getEFOrientation(endEffector1)
        oriSteps1 = np.linspace(currentOrientation1, orientation1, len(traj1))
        if endEffector2 != None:
            endEffector2Position = self.getEFLocationAndOrientation(endEffector2)[0]
            steps2 = np.linspace(endEffector2Position, targetPosition2, interpolationSteps)
            currentOrientation2 = self.getEFOrientation(endEffector2)
            if trajectory2.size != 0:
                traj2 = trajectory2
            else:
                traj2 = steps2

            oriSteps2 = np.linspace(currentOrientation2, orientation2, len(traj2))
        
        pltTime = [0]
        pltDistance = [dist]
        startingTime = time.time()
        
        for i in range(1, len(traj1)):
            prev_q = np.copy(current_q)
            if incrementalOrientation == True:
                targetOrientation = oriSteps1[i]
            else:
                targetOrientation = orientation1
            dTheta1 = self.inverseKinematics(endEffector1, traj1[i], targetOrientation, noChest)
            current_q += dTheta1
            if endEffector2 != None:
                if incrementalOrientation == True:
                    targetOrientation = oriSteps2[i]
                else:
                    targetOrientation = orientation2
                dTheta2 = self.inverseKinematics(endEffector2, traj2[i], targetOrientation, noChest)
                current_q += dTheta2
            for j in range(len(jointNames)):
                self.jointTargetPos[jointNames[j]] = current_q[j]
                jointTarg = current_q[j]
                jointPos = super().getJointPos(jointNames[j])
                self.prevErrors[jointNames[j]] = jointTarg - jointPos
            for j in range(200):
                self.tick()
                
            endEffectorPosition = self.getEFLocationAndOrientation(endEffector1)[0]
            currentTime = time.time() - startingTime
            pltTime.append(currentTime)
            distanceToTarget = np.linalg.norm(targetPosition1 - endEffectorPosition)
            pltDistance.append(distanceToTarget)
            if distanceToTarget < threshold:
                break
        return pltTime, pltDistance


    def move_chest(self, traj):
        for i in range(len(traj)):
            self.jointTargetPos['CHEST_JOINT0'] = traj[i]
            for j in range(200):
                self.tick()
        
    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control.
        new_prevErrors = {}
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            torque = 0

            if bool(self.prevErrors) == True:
                targetPos = self.jointTargetPos[joint]
                currentPos = super().getJointPos(joint)
                torque = self.calculateTorque(targetPos, currentPos, self.prevErrors[joint], 0, kp, ki, kd)
                self.prevErrors[joint] = targetPos - currentPos

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        xs = points[:, 0]
        ys = points[:, 1]
        cs = CubicSpline(xs, ys)
        start_x = points[0][0]
        end_x = points[-1][0]
        xs_nTimes = np.linspace(start_x, end_x, nTimes)
        spline_points = np.empty((0, 2), float)
        for i in range(nTimes):
            spline_x = xs_nTimes[i]
            spline_y = cs(spline_x)
            splinePoint = np.array([spline_x, spline_y])
            spline_points = np.append(spline_points, np.array([splinePoint]), axis=0)

        return spline_points

    def generate_traj(self, task, points, nTimes=20):
        if task == 'push':
            xypoints = points[:, 0:2]
            reverse_xypoints = np.flip(xypoints, 0)
            reverse_curve = self.cubic_interpolation(reverse_xypoints, nTimes)
            xy_traj = np.flip(reverse_curve, 0)
            zpoints = points[:, 2]
            z_traj = np.linspace(zpoints[0], zpoints[-1], nTimes)
            traj = np.empty((0, 3), float)
            for i in range(nTimes):
                point_3d = np.append(xy_traj[i], z_traj[i])
                traj = np.append(traj, np.array([point_3d]), axis=0)
        elif task == 'push2':
            xypoints = points[:, 0:2]
            xy_traj = self.cubic_interpolation(xypoints, nTimes)
            zpoints = points[:, 2]
            z_traj = np.linspace(zpoints[0], zpoints[-1], nTimes)
            traj = np.empty((0, 3), float)
            for i in range(nTimes):
                point_3d = np.append(xy_traj[i], z_traj[i])
                traj = np.append(traj, np.array([point_3d]), axis=0)
        elif task == 'grasp1':
            yzpoints = points[:, 1:3]
            reverse_yzpoints = np.flip(yzpoints, 0)
            reverse_curve = self.cubic_interpolation(reverse_yzpoints, nTimes)
            yz_traj = np.flip(reverse_curve, 0)
            xpoints = points[:, 0]
            x_traj = np.linspace(xpoints[0], xpoints[-1], nTimes)
            traj = np.empty((0, 3), float)
            for i in range(nTimes):
                point_3d = np.concatenate((np.array([x_traj[i]]), yz_traj[i]))
                traj = np.append(traj, np.array([point_3d]), axis=0)
        elif task == 'grasp2':
            yzpoints = points[:, 1:3]
            yz_traj = self.cubic_interpolation(yzpoints, nTimes)
            xpoints = points[:, 0]
            x_traj = np.linspace(xpoints[0], xpoints[-1], nTimes)
            traj = np.empty((0, 3), float)
            for i in range(nTimes):
                point_3d = np.concatenate((np.array([x_traj[i]]), yz_traj[i]))
                traj = np.append(traj, np.array([point_3d]), axis=0)
                
        return traj

    def pushCube(self, cubePos, targetPos):
        endEffector = 'LHAND'
        endEffectorPos = self.getEFLocationAndOrientation(endEffector)[0]
        #half x-size of cube
        offsetCubePos = cubePos - np.array([0.035, 0, 0])
        #half x-size of lhand
        offsetCubePos -= np.array([0.0285, 0, 0])
        behindCubePos = cubePos - np.array([0.08, 0, 0])
        behindCubePos += np.array([0, 0.04, 0])
        rightAnglePoint = np.array([cubePos[0], endEffectorPos[1], endEffectorPos[2]])
        yHalfway = ((rightAnglePoint - cubePos)/2)[1]
        thirdPoint = np.array([0.2, yHalfway, (endEffectorPos[2]+cubePos[2])/2])
        speed = 0.018673746031553178
        multiplierForSteps = 3.5
        dist1 = np.linalg.norm(thirdPoint - endEffectorPos)
        steps1 = int((dist1/speed)*multiplierForSteps)
        points1 = np.array([endEffectorPos, rightAnglePoint, thirdPoint])
        traj = self.generate_traj('push', points1, nTimes=steps1)
        offsetCubePosHigher = np.array([offsetCubePos[0], offsetCubePos[1], thirdPoint[2]])
        dist2 = np.linalg.norm(offsetCubePosHigher - thirdPoint)
        steps2 = int((dist2/speed)*multiplierForSteps)
        points2 = np.array([thirdPoint, behindCubePos, offsetCubePosHigher])
        traj2 = self.generate_traj('push2', points2, nTimes=steps2)
        traj2 = traj2[1:]
        traj = np.concatenate((traj, traj2), axis=0)
        dist3 = np.linalg.norm(targetPos - offsetCubePosHigher)
        steps3 = int((dist3/speed)*multiplierForSteps)
        targetPos = np.array([targetPos[0], targetPos[1], thirdPoint[2]])
        finalTraj = np.linspace(offsetCubePosHigher, targetPos, steps3)
        finalTraj = finalTraj[1:]
        traj = np.concatenate((traj, finalTraj), axis=0)
        targetOrientation = np.zeros(3)
        self.move_with_PD(endEffector, np.array([0, 0, 0]), speed=0.05, orientation1=targetOrientation, threshold=1e-3, maxIter=3000, debug=False, verbose=False, trajectory1=traj)


    def grasp(self, cubePos, targetPos):
        endEffector1 = 'LHAND'
        endEffector1Pos = self.getEFLocationAndOrientation(endEffector1)[0]
        cubePos -= np.array([0, 0, 0.06])
        cubeOffset1 = cubePos + np.array([0, 0.098, 0])
        thirdPoint1 = (cubeOffset1 - endEffector1Pos)/2 + endEffector1Pos
        thirdPoint1 += np.array([0, 0, 0.01])
        points1 = np.array([endEffector1Pos, thirdPoint1, cubeOffset1])
        traj1 = self.generate_traj('grasp1', points1, nTimes=70)
        ori1 = np.array([0, 0, np.deg2rad(-90)])
        endEffector2 = 'RHAND'
        endEffector2Pos = self.getEFLocationAndOrientation(endEffector2)[0]
        cubeOffset2 = cubePos - np.array([0, 0.098, 0])
        thirdPoint2 = (cubeOffset2 - endEffector2Pos)/2 + endEffector2Pos
        thirdPoint2 += np.array([0, 0, 0.01])
        points2 = np.array([endEffector2Pos, thirdPoint2, cubeOffset2])
        traj2 = self.generate_traj('grasp2', points2, nTimes=70)
        ori1 = np.array([0, 0, np.deg2rad(-90)])
        ori2 = np.array([0, 0, np.deg2rad(90)])
        self.move_with_PD(endEffector1, np.array([0, 0, 0]), endEffector2, np.array([0, 0, 0]), speed=0.05, orientation1=ori1, orientation2=ori2, threshold=1e-3, maxIter=3000, debug=False, verbose=False, trajectory1=traj1, trajectory2=traj2, noChest=True, incrementalOrientation=True)
        time.sleep(1)
        endEffector1Pos = self.getEFLocationAndOrientation(endEffector1)[0]
        targetPos1 = endEffector1Pos + np.array([0.1, 0, 0.13])
        traj1 = np.linspace(endEffector1Pos, targetPos1, 70)
        endEffector2Pos = self.getEFLocationAndOrientation(endEffector2)[0]
        targetPos2 = endEffector2Pos + np.array([0.1, 0, 0.13])
        traj2 = np.linspace(endEffector2Pos, targetPos2, 70)
        self.move_with_PD(endEffector1, np.array([0, 0, 0]), endEffector2, np.array([0, 0, 0]), speed=0.05, orientation1=ori1, orientation2=ori2, threshold=1e-3, maxIter=3000, debug=False, verbose=True, trajectory1=traj1, trajectory2=traj2, noChest=True, incrementalOrientation=True)
        time.sleep(1)
        halfTarget = 0.106568
        centreToCorner = math.sqrt((halfTarget**2)+(halfTarget**2))
        corner1 = targetPos - np.array([centreToCorner, 0, 0])
        corner2 = targetPos + np.array([0, centreToCorner, 0])
        betweenCorners1 = (corner2 - corner1)/2 + corner1
        corner3 = targetPos - np.array([0, centreToCorner, 0])
        corner4 = targetPos + np.array([centreToCorner, 0, 0])
        betweenCorners2 = (corner4 - corner3)/2 + corner3
        ori1 = np.array([0, 0, np.deg2rad(-47)])
        ori2 = np.array([0, 0, np.deg2rad(90+47)])
        endEffector1Pos = self.getEFLocationAndOrientation(endEffector1)[0]
        endEffector2Pos = self.getEFLocationAndOrientation(endEffector2)[0]
        hTargetPos1 = np.array([betweenCorners1[0], betweenCorners1[1], endEffector1Pos[2]])
        hTargetPos2 = np.array([betweenCorners2[0], betweenCorners2[1], endEffector2Pos[2]])
        chestTraj1 = np.linspace(endEffector1Pos, hTargetPos1, 150)
        chestTraj2 = np.linspace(endEffector2Pos, hTargetPos2, 150)
        chestPos = super().getJointPos('CHEST_JOINT0')
        chestTraj = np.linspace(chestPos, np.deg2rad(47), 250)
        self.move_chest(chestTraj)
        time.sleep(1)
        endEffector1Pos = self.getEFLocationAndOrientation(endEffector1)[0]
        endEffector2Pos = self.getEFLocationAndOrientation(endEffector2)[0]
        traj1 = np.linspace(endEffector1Pos, betweenCorners1, 70)
        traj2 = np.linspace(endEffector2Pos, betweenCorners2, 70)
        self.move_with_PD(endEffector1, np.array([0, 0, 0]), endEffector2, np.array([0, 0, 0]), speed=0.05, orientation1=ori1, orientation2=ori2, threshold=1e-3, maxIter=3000, debug=False, verbose=True, trajectory1=traj1, trajectory2=traj2, noChest=True, incrementalOrientation=True)
