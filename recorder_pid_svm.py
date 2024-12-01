import numpy as np
import gymnasium as gym
from dataset.envs.reacher_v6 import ReacherEnv  # Assumes ReacherEnv is already implemented
from dataset.envs.reacher3_v6 import Reacher3Env
from dataset.envs.marrtino_arm import MARRtinoArmEnv
from sklearn.multioutput import MultiOutputRegressor
from jacobian_svm import IK
from joblib import load
import warnings

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names")

robot = 'r2'  # r3 or r5

# Paths
model_path = 'models/svm/svm_' + robot + '.joblib'


def get_env(robot=''):
    if robot == 'r2':
        return ReacherEnv(render_mode="rgb_array")
    elif robot == 'r3':
        return Reacher3Env(render_mode="rgb_array")
    elif robot == 'r5':
        return MARRtinoArmEnv(render_mode="rgb_array")
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
        self.model = load(model_path)

        # Get the input and output dimensions of the model
        self.input_dim = self.model.n_features_in_
        self.output_dim = len(self.model.estimators_)

        self.pid_controllers = [PIDController(Kp=1, Ki=0.01, Kd=0.1, dt=self.dt) for _ in range(env.njoints)]

    def compute_joint_angles(self, target_position, verbose=False):

        # IK model predicts joint angles from a target position
        target_position = np.array(target_position)

        if robot == 'r2' or robot == 'r3':
            target_position = np.concatenate((target_position, np.zeros(2)), axis=0)  # Add the 2D quaternion
        elif robot == 'r5':
            target_position = np.concatenate((target_position, np.zeros(4)), axis=0)  # Add the 3D quaternion

        # Add trigonometric values to the initial joint for the model
        init_joints = np.zeros(self.env.njoints)
        init_joints = np.concatenate((init_joints, np.cos(init_joints), np.sin(init_joints)), axis=0)

        # Call the IK function
        target_joints = IK(self.model, init_joints, target_position, eta=0.01, orientation=False, verbose=verbose)

        # Normalize in -pi, pi the target joints
        target_joints = np.array([((j + np.pi) % (2 * np.pi)) - np.pi for j in target_joints])

        return target_joints[:self.env.njoints]

    def apply_control(self, desired_joints, max_steps=200, video_path="videos/simulation.mp4"):
        # Record a video using the gymnasium wrappers
        from gymnasium.wrappers import RecordVideo

        # Wrap the environment with the video recorder
        env = RecordVideo(self.env, video_path)

        # Reset the environment
        observation, _ = env.reset()
        for step in range(max_steps):
            # Get current joint angles from the observation
            current_joints = observation[:self.env.njoints]

            # Compute control actions using PID
            control_actions = np.array([
                self.pid_controllers[i].compute(desired_joints[i], current_joints[i])
                for i in range(self.env.njoints)
            ])

            # Apply the control actions to the environment
            observation, _, _, _, _ = env.step(control_actions)

            # Render the environment for visualization
            self.env.render()
            error = np.linalg.norm(desired_joints - current_joints)

            # Check if the robot reached the target position
            if np.allclose(current_joints, desired_joints, atol=1e-2):
                print("Target position reached!")
                # break  # Do not stop because it can overshoot
            else:
                print(f"Step {step}: Current Joints: {current_joints}, Target Joints: {desired_joints}, Error: {error}")

        # After reaching the target, keep the environment open
        print("Simulation complete. Keeping environment open.")
        env.close()


if __name__ == "__main__":
    # Create the environment
    env = get_env(robot)

    # Initialize the robot controller
    robot_controller = RobotController(env, model_path, dt=0.01)

    target_joints = []
    if robot == 'r5':
        target_joints = np.array([0.403, 1.293, -1.573, 1.119, -1.063])  # Example values
    elif robot == 'r3':
        target_joints = np.array([0.403, 1.293, -1.573])
    elif robot == 'r2':
        target_joints = np.array([0.403, 1.293])

    print("Target Joints: ", target_joints)

    # Apply the control to reach the target position and record the video
    robot_controller.apply_control(target_joints, video_path="videos/simulation_" + robot + ".mp4")
