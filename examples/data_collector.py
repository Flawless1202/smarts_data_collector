import importlib
import logging
import time
import os
import random
import shutil

import gym
import numpy as np
from scipy.spatial.transform import Rotation
from examples import default_argument_parser

from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

OUTPUT_DIR = "output_dataset"

AGENT_ID = "Agent-007"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 960

perpetual_rider_agent = importlib.import_module("zoo.policies.perpetual_rider_agent")


class RandomColor:

    def __init__(self, seed=0):
        self.rng = random.Random()
        self.seed = seed
        self.reset()

    def reset(self):
        self.rng.seed(self.seed)

    def __call__(self):
        return self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255)


def make_camera_pose():
    rot = Rotation.from_euler('zxy', [90, 90, 90], degrees=True).as_rotvec().flatten().tolist()
    return np.array(rot + [0., 0., 0.])


def Rt2T(R, t):
    Rt = np.concatenate((R.reshape((3, 3)), t.reshape((3, 1))), axis=1)
    T = np.concatenate((Rt, np.array([0, 0, 0, 1]).reshape((1, 4))), axis=0)
    return T

def mkdir(path):
    os.makedirs(path, exist_ok=True)


def rm(path):
    if not os.path.exists(path): return
    if os.path.isfile(path): os.remove(path)
    elif os.path.isdir(path): shutil.rmtree(path)


def project(ego_vehicle_pose, other_vehicle_pose, camera_pose, camera_intrinsic, other_vehicle_size, frame, color):

    Rwo = Rotation.from_rotvec(other_vehicle_pose[:3]).as_matrix()
    Rws = Rotation.from_rotvec(ego_vehicle_pose[:3]).as_matrix()
    two = other_vehicle_pose[3:]
    tws = ego_vehicle_pose[3:]
    Two = Rt2T(Rwo, two)
    Tws = Rt2T(Rws, tws)
    Tso = np.linalg.inv(Tws) @ Two

    Rcs = Rotation.from_rotvec(camera_pose[:3]).as_matrix()
    tcs = camera_pose[3:]

    half_lwh = np.array(other_vehicle_size.as_lwh, dtype=np.float).reshape((1, 3))[:, [1,0,2]] * 1.5 / 2
    points_other_vehicle = half_lwh * np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1],
                                                [1, -1, -1], [-1, 1, 1], [-1, 1, -1],
                                                [-1, -1, 1], [-1, -1, -1]], dtype=np.float)

    points_camera = Rcs.reshape((1, 3, 3)) @ (
            Tso[:3, :3].reshape(1, 3, 3) @ points_other_vehicle.reshape((8, 3, 1)) + Tso[:3, 3:].reshape(1, 3, 1)
            ) + tcs.reshape((1, 3, 1))

    z = points_camera[:, 2:3, :]
    mask = (np.logical_and(z > 1, z < 100)).flatten().tolist()
    if sum(mask) != 8:
        return frame, None

    points_camera /= z

    points_image = camera_intrinsic @ points_camera

    x1, y1 = points_image[:, 0:1].min(), points_image[:, 1:2].min()
    x2, y2 = points_image[:, 0:1].max(), points_image[:, 1:2].max()
    xyxy = None if x1 > FRAME_WIDTH or x2 < 0 or y1 > FRAME_HEIGHT or y2 < 0 else (x1, y1, x2, y2)
    return frame, xyxy


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed, max_episode_steps=max_episode_steps
        ),
        agent_builder=perpetual_rider_agent.PerpetualRiderAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=True,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
    )

    camera_pose = make_camera_pose()
    camera_intrinsic = np.array((250.0, 0.0, FRAME_WIDTH/2, 0.0, 250.0, FRAME_HEIGHT/2)).reshape((1, 2, 3))
    color_rng = RandomColor(10)

    scene_idx = 7001
    end_scene_idx = 8001

    rm(f"{OUTPUT_DIR}")
    mkdir(f"{OUTPUT_DIR}/annotations/")
    mkdir(f"{OUTPUT_DIR}/ego_poses/")
    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)
        episode_sim_time_epoch = 0
        episode_sim_time_frame_with_visible_object = 0

        mkdir(f"{OUTPUT_DIR}/frames/scene-{scene_idx:04d}/")
        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)
            timestamp = episode.start_time + episode.sim_time

            # at most 18 seconds total
            # if episode.sim_time < 2.:
            #     continue
            if episode.sim_time > 31.99:
                scene_idx += 1
                break

            # 10 seconds for each scene
            if episode.sim_time - episode_sim_time_epoch > 9.99:
                scene_idx += 1
                episode_sim_time_epoch = episode.sim_time
                mkdir(f"{OUTPUT_DIR}/frames/scene-{scene_idx:04d}/")

            # generate ego_poses
            ego_rot_quat = Rotation.from_euler('z', agent_obs.ego_vehicle_state.heading, degrees=False).as_quat().flatten()
            ego_translate = agent_obs.ego_vehicle_state.position.flatten()
            ego_pose = ', '.join([str(x) for x in np.concatenate((ego_rot_quat, ego_translate)).tolist()])
            with open(f'{OUTPUT_DIR}/ego_poses/scene-{scene_idx:04d}_ego_pose.csv', 'a') as ego_pose_file:
                ego_pose_file.write(f'{timestamp}, {ego_pose}\n')

            # generate frame
            frame_ego = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            ego_vehicle_pose = np.array((0, 0, agent_obs.ego_vehicle_state.heading, *agent_obs.ego_vehicle_state.position))
            color_rng.reset()
            visible_object_counter = 0
            for object_uid, neighborhood_vehicle_state in enumerate(agent_obs.neighborhood_vehicle_states):
                other_vehicle_pose = np.array((0, 0, neighborhood_vehicle_state.heading, *neighborhood_vehicle_state.position))
                other_vehicle_size = neighborhood_vehicle_state.bounding_box
                color = color_rng()
                frame_ego, xyxy = project(ego_vehicle_pose, other_vehicle_pose,
                                          camera_pose, camera_intrinsic,
                                          other_vehicle_size, frame_ego,
                                          color)
                # generate annotations
                if xyxy is not None:
                    with open(f'{OUTPUT_DIR}/annotations/scene-{scene_idx:04d}_instances_ann.csv', 'a') as annotation_file:
                        annotation_file.write(f"{timestamp}, {object_uid}, " + ", ".join([str(x) for x in xyxy]) + "\n")
                    visible_object_counter += 1

            # remove a scene with large blank

            if visible_object_counter < 1:
                if episode.sim_time - episode_sim_time_frame_with_visible_object > 0.5:
                    break
            else:
                episode_sim_time_frame_with_visible_object = episode.sim_time

        # remove scenes less than 6 seconds
        if episode.sim_time - episode_sim_time_epoch < 9.99:
            rm(f"{OUTPUT_DIR}/frames/scene-{scene_idx:04d}/")
            rm(f'{OUTPUT_DIR}/annotations/scene-{scene_idx:04d}_instances_ann.csv')
            rm(f'{OUTPUT_DIR}/ego_poses/scene-{scene_idx:04d}_ego_pose.csv')

        time.sleep(2)
        if scene_idx >= end_scene_idx:
            break

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("data-collector-agent")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
