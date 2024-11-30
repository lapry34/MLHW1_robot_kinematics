import numpy as np
import gymnasium as gym
from dataset.envs.reacher_v6 import ReacherEnv  # Assumes ReacherEnv is already implemented
from dataset.envs.reacher3_v6 import Reacher3Env
from dataset.envs.marrtino_arm import MARRtinoArmEnv
from jacobian_NN import IK
import tensorflow as tf
from tensorflow import keras

#set robot to r2, r3, or r5
robot = 'r5' 
tuned = True

# Paths
model_path = ''

#choose the model path
if tuned:
    model_path = 'models/NN/model_' + robot + '_tuned.keras'
else:
    model_path = 'models/NN/model_' + robot + '.keras'


#force tensorflow to use CPU
tf.config.set_visible_devices([], 'GPU')

def get_env(robot=''):
    if robot == 'r2':
        return ReacherEnv(render_mode="human")
    elif robot == 'r3':
        return Reacher3Env(render_mode="human")
    elif robot == 'r5':
        return MARRtinoArmEnv(render_mode="human")
    else:
        return None

# PID Controller
class PIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.previous_error = 0
        self.integral = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error * self.dt
        derivative = (error - self.previous_error) / self.dt
        self.previous_error = error
        control_action = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        return control_action


# Robot Controller
class RobotController:
    def __init__(self, env, model_path, dt=0.01):
        self.env = env
        self.dt = dt
        self.model = keras.models.load_model(model_path)

        # Get the input and output dimensions of the model
        self.input_dim = self.model.input_shape[1]
        self.output_dim = self.model.output_shape[1]

        self.pid_controllers = [PIDController(Kp=1, Ki=0.01, Kd=0.1, dt=self.dt) for _ in range(env.njoints)]

    def compute_joint_angles(self, target_position, verbose=False):

        # IK model predicts joint angles from a target position
        target_position = np.array(target_position)

        if robot=='r2' or robot=='r3':
            target_position = np.concatenate((target_position, np.zeros(2)), axis=0) #we add the 2D quaternion
        elif robot=='r5':
            target_position = np.concatenate((target_position, np.zeros(4)), axis=0) #we add the 3D quaternion
        
        # Add trigonometric values to the initial joint for the model
        init_joints = np.zeros(self.env.njoints)
        init_joints = np.concatenate((init_joints, np.cos(init_joints), np.sin(init_joints)), axis=0)

        # Call the IK function
        target_joints = IK(self.model, init_joints, target_position, orientation=False, verbose=verbose)

        #normalize in -pi, pi the target joints
        target_joints = np.array([((j + np.pi) % (2 * np.pi)) - np.pi for j in target_joints])

        return target_joints[:self.env.njoints]

    def apply_control(self, desired_joints, max_steps=200):
        # Reset the environment
        observation, _ = self.env.reset()
        for step in range(max_steps):
            # Get current joint angles from the observation
            current_joints = observation[:self.env.njoints]

            # Compute control actions using PID
            control_actions = np.array([
                self.pid_controllers[i].compute(desired_joints[i], current_joints[i])
                for i in range(self.env.njoints)
            ])

            # Apply the control actions to the environment
            observation, _, _, _, _ = self.env.step(control_actions)

            # Render the environment for visualization
            self.env.render()
            error = np.linalg.norm(desired_joints - current_joints)
            
            # Check if the robot reached the target position
            if np.allclose(current_joints, desired_joints, atol=1e-2):
                print("Target position reached!")
                #break #we dont stop because it can overshoot
            else:
                print(f"Step {step}: Current Joints: {current_joints}, Target Joints: {desired_joints}, Error: {error}")
            

        # After reaching the target, keep simulation running
        print("Simulation complete. Keeping environment open.")
        while True:
            self.env.render()
            self.env.step(np.zeros(self.env.njoints))

if __name__ == "__main__":
    # Create the environment
    env = get_env(robot)

    # Initialize the robot controller
    robot_controller = RobotController(env, model_path, dt=0.01)

    # Set a target position (e.g., [0.1, 0.1] for 2D)
    target_position = [0.05, 0.07]

    if env.njoints == 5: #R5 robot
        target_position.append(0.2)

    #use the IK to get the joint angles
    print("Using Inverse Kinematics to compute target joints...")
    target_joints = robot_controller.compute_joint_angles(target_position)
    print("Target Joints: ", target_joints)

    # Apply the control to reach the target position
    robot_controller.apply_control(target_joints)
