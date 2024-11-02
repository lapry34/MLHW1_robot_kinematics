import sys, time
sys.path.append("..")
import argparse
import gymnasium as gym

from envs.reacher_v6 import ReacherEnv
from envs.reacher3_v6 import Reacher3Env
from envs.marrtino_arm import MARRtinoArmEnv

def print_header(njoints):
    for i in range(njoints):
        print(f"cos(j{i});")
    for i in range(njoints):
        print(f"sin(j{i});")
    print("ee_x;ee_y;ee_qw;ee_qz")

def print_obs(obs, njoints):
    # cosx(j..), sin(j..)
    r = ""
    for i in range(0,2*njoints):
        r = r + f"{obs[i]:6.3f}; "
    # fingertip pos
    for i in range(-4,-2):
        r = r + f"{obs[i]:6.3f}; "
    # fingertip quat
    for i in range(-2,0):
        r = r + f"{obs[i]:6.3f}; "
    print(r[0:-2])

def dorun(args):

    render_mode="human" if args.render else None
        
    if args.env=='r2':
        env = ReacherEnv(render_mode=render_mode)
    elif args.env=='r3':
        env = Reacher3Env(render_mode=render_mode)
    elif args.env=='r5':
        env = MARRtinoArmEnv(render_mode=render_mode)
    else:
        print(f"Unknown environment {args.env}")
        sys.exit(1)

    #print(f"Observation: {env.observation_space}")
    #print(f"Action: {env.action_space}")

    if args.log:
        print_header(env.njoints)

    observation, info = env.reset(seed=args.seed)
    env.action_space.seed(seed=args.seed)
    if args.log:
        print_obs(observation, env.njoints)

    for _ in range(1,args.steps):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)
        if args.log:
            print_obs(observation, env.njoints)
        if render_mode=="human":
            time.sleep(0.1)
        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-env", type=str, default="r2",
        help="environment [r2,r3,r5] (default: r2)")
    parser.add_argument("-steps", type=int, default=10000,
        help="Execution steps (default: 10,000)")
    parser.add_argument("-seed", type=int, default=1000,
        help="Random seed (default: 1000)")
    parser.add_argument('--render', default = False, action ='store_true',
        help='Enable rendering')
    parser.add_argument('--log', default = False, action ='store_true',
        help='Enable data log')

    args = parser.parse_args()
    dorun(args)



