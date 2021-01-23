import math

import numpy as np
from transitions import Machine

from smarts.core.agent import Agent
from smarts.core.sensors import Observation


class PerpetualRiderAgent(Agent):
    lateral_states = ['keep_lane', 'change_lane']

    def __init__(self, desired_speed=7.5, radical=0.75, minimal_ttc=5.):
        super().__init__()
        self.lateral_decider = Machine(model=self, states=PerpetualRiderAgent.lateral_states, initial='keep_lane')
        self.lateral_decider.add_transition("decide", "keep_lane", "change_lane",
                                            conditions=['am_too_slow', 'can_safely_change_lane'])
        self.lateral_decider.add_transition("decide", "keep_lane", "keep_lane", after='act_keep_lane')
        self.lateral_decider.add_transition("decide", "change_lane", "keep_lane",
                                            conditions=['changing_seems_finished'], after='act_keep_lane')
        self.lateral_decider.add_transition("decide", "change_lane", "change_lane")

        self.desired_speed = desired_speed
        self.radical = radical
        self.minimal_ttc = minimal_ttc

    @property
    def am_too_slow(self):
        return self.ego_speed < self.desired_speed * self.radical

    @property
    def can_safely_change_lane(self):
        if self.neighbors['ego_lane_front'] is None:
            return False

        for waypoints in self.obs.waypoint_paths:
            if waypoints[0].lane_index + 1 == self.ego_lane_index:
                if self.calc_ttc(self.neighbors['left_lane_front']) <= \
                        self.calc_ttc(self.neighbors['ego_lane_front']):
                    continue
                safe_speed_min = max(0, self.safe_speed(self.neighbors['left_lane_behind']))
                safe_speed_max = self.safe_speed(self.neighbors['left_lane_front'])
                if safe_speed_min > safe_speed_max:
                    continue
                self.ego_lane_id_before = self.obs.ego_vehicle_state.lane_id
                self.command = ((safe_speed_min + safe_speed_max) / 2., 1)
                return True
            if waypoints[0].lane_index - 1 == self.ego_lane_index:
                if self.calc_ttc(self.neighbors['right_lane_front']) <= \
                        self.calc_ttc(self.neighbors['ego_lane_front']):
                    continue
                safe_speed_min = max(0, self.safe_speed(self.neighbors['right_lane_behind']))
                safe_speed_max = self.safe_speed(self.neighbors['right_lane_front'])
                if safe_speed_min > safe_speed_max:
                    continue
                self.ego_lane_id_before = self.obs.ego_vehicle_state.lane_id
                self.command = ((safe_speed_min + safe_speed_max) / 2., -1)
                return True
        return False

    @property
    def changing_seems_finished(self):
        if self.ego_lane_id_before != self.obs.ego_vehicle_state.lane_id:
            return True
        return False

    def act_keep_lane(self):
        all_safe_speed = [self.desired_speed]

        safe_speed = self.desired_speed
        for ne in self.obs.neighborhood_vehicle_states:
            if self.afront_of_me(ne) and self.dist(ne) < 80:
                safe_speed = min(self.safe_speed(ne), safe_speed)
                all_safe_speed.append(safe_speed)

        self.command = (safe_speed, 0)

        min_dist = math.inf
        nearest_ne = None
        for ne in self.obs.neighborhood_vehicle_states:
            if self.dist(ne) < min_dist:
                nearest_ne = ne
                min_dist = self.dist(nearest_ne)

    def afront_of_me(self, ne):
        rel_pos = ne.position - self.ego_position
        rel_ang_world = math.atan2(rel_pos[1].item(), rel_pos[0].item())
        rel_ang_self = rel_ang_world - self.obs.ego_vehicle_state.heading

        return math.sin(rel_ang_self) > 0

    def dist(self, ne):
        if ne is None: return float('inf')
        rel_pos = ne.position - self.ego_position
        dist_ = np.linalg.norm(rel_pos, ord=2)
        return dist_

    def calc_ttc(self, ne):
        if ne is None: return float('inf')
        if self.afront_of_me(ne):
            ttc = self.dist(ne) / (self.ego_speed - ne.speed)
        else:
            ttc = self.dist(ne) / (ne.speed - self.ego_speed)
        return ttc

    def safe_speed(self, ne):
        # print(f"ne is None: {ne is None}, ", end="\n" if ne is None else "")
        if ne is None: return self.desired_speed
        if self.afront_of_me(ne):
            # print("ne is afront of me")
            ssp = self.dist(ne) / self.minimal_ttc + ne.speed
        else:
            # print("ne is behind me")
            ssp = - self.dist(ne) / self.minimal_ttc + ne.speed
        return ssp

    def parse_obs(self):
        self.ego_speed = self.obs.ego_vehicle_state.speed
        self.ego_lane_index = self.obs.ego_vehicle_state.lane_index
        self.ego_position = self.obs.ego_vehicle_state.position

        self.neighbors = dict(
            ego_lane_front=None,
            left_lane_front=None,
            left_lane_behind=None,
            right_lane_front=None,
            right_lane_behind=None)

        def update_neighbors(key, ne):
            if self.neighbors[key] is None or self.dist(ne) < self.dist(self.neighbors[key]):
                self.neighbors[key] = ne

        for ne in self.obs.neighborhood_vehicle_states:
            if self.ego_lane_index == ne.lane_index and self.afront_of_me(ne):
                update_neighbors('ego_lane_front', ne)
            elif self.ego_lane_index - 1 == ne.lane_index:
                if self.afront_of_me(ne):
                    update_neighbors('left_lane_front', ne)
                else:
                    update_neighbors('left_lane_behind', ne)
            elif self.ego_lane_index + 1 == ne.lane_index:
                if self.afront_of_me(ne):
                    update_neighbors('right_lane_front', ne)
                else:
                    update_neighbors('right_lane_behind', ne)
            else:
                continue

    def act(self, obs: Observation):
        self.obs = obs
        self.parse_obs()
        self.decide()
        return self.command
