from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
    A task in which the agent must perform steady, level flight maintaining its initial heading.
    Every 150 sec a new target heading is set.
"""

class HeadingControlTask(Task):

    state_var = [
      c.delta_altitude,
      c.delta_heading,
      c.velocities_v_down_fps,
      c.velocities_vc_fps,
      c.velocities_p_rad_sec,
      c.velocities_q_rad_sec,
      c.velocities_r_rad_sec
    ]

    action_var = [
      c.fcs_aileron_cmd_norm,
      c.fcs_elevator_cmd_norm,
      c.fcs_throttle_cmd_norm,
      c.fcs_rudder_cmd_norm,
    ]

    init_conditions = {
      c.ic_h_sl_ft: 10000,
      c.ic_terrain_elevation_ft: 0,
      c.ic_long_gc_deg: 1.442031,
      c.ic_lat_geod_deg: 43.607181,
      c.ic_u_fps: 800,
      c.ic_v_fps: 0,
      c.ic_w_fps: 0,
      c.ic_p_rad_sec: 0,
      c.ic_q_rad_sec: 0,
      c.ic_r_rad_sec: 0,
      c.ic_roc_fpm: 0,
      c.ic_psi_true_deg: 100,
      c.target_heading_deg: 100,
      c.target_altitude_ft: 10000,
      c.fcs_throttle_cmd_norm: 0.8,
      c.fcs_mixture_cmd_norm: 1,
      c.gear_gear_pos_norm : 0,
      c.gear_gear_cmd_norm: 0,
      c.steady_flight:150
    }

    def __init__(self, floatingAction=True):
       super().__init__()
       # Variables we want to track and output at render time:
       self.mostRecentRewards = {}
       self.stopReason = None
       self.otherInfo = None
       self.simStopInfo = None
       self.numTargetChanges = 0

       # Debug to count:
       self.stepCount = 0
       self.simTime = 0

       # How screwed is too screwed?
       self.worstCaseAltitudeDelta = 2000

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in HeadingControlTask.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       print(f"Prop min/maxes:\n{self.observation_minMaxes}")
       # The deltaAltitude is 40k.  Since we'll limit our aircraft to a
       # delta-altitude of 5k, change that variable:
       self.observation_minMaxes[0] = [-2000, 2000]

       # Assume floating point:
       # All actions are [-1, 1] except throttle which goes [0, 0.9]:
       fullActionSpace = spaces.Box(low=np.array([-1.0, -1.0, 0, -1.0]),
                                      high=np.array([1.0, 1.0, 0.9, 1.0]),
                                      dtype=np.float32)

       zeroRudderActionSpace = spaces.Box(low=np.array([-1.0, -1.0, 0]),
                                      high=np.array([1.0, 1.0, 0.9]),
                                      dtype=np.float32)

       self.action_space = zeroRudderActionSpace

       # Bangbang controls have different input requirements:
       if not self.floatingAction:
          self.action_space = spaces.Discrete(18)

       self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(7,),
                                           dtype=np.float32)

    # Need to override get_observation_space and get_action_space:
    def get_observation_space(self):
       return self.observation_space

    def get_action_space(self):
       return self.action_space

    def convertObservation(self, observation):
       # print(f"Starting observation: {observation}")
       retObs = np.array(observation).reshape(-1)

       for i in range(len(retObs)):
          retObs[i] = (((retObs[i] - self.observation_minMaxes[i][0]) / (self.observation_minMaxes[i][1] - self.observation_minMaxes[i][0])) - 0.5) * 2.0

          # Make sure we stay within [-1, 1]:
          retObs[i] = min(1.0, max(-1.0, retObs[i]))

       # print(f"Ending observation{retObs}")
       return retObs


    def reset(self):
       # Variables we want to track and output at render time:
       self.mostRecentRewards = {}
       self.stopReason = None
       self.otherInfo = None
       self.simStopInfo = None
       self.numTargetChanges = 0

       # Debug to count:
       self.stepCount = 0
       self.simTime = 0

    def get_reward(self, state, sim):
        '''
        Reward according to altitude, heading, and axes accelerations.
        '''
        return self.get_reward_maintain_altitude(state, sim)

        # heading_r = math.exp(-math.fabs(sim.get_property_value(c.delta_heading)))
        # alt_r = math.exp(-math.fabs(sim.get_property_value(c.delta_altitude)))
        # angle_speed_r = math.exp(-(0.1*math.fabs(sim.get_property_value(c.accelerations_a_pilot_x_ft_sec2)) +
        #                         0.1*math.fabs(sim.get_property_value(c.accelerations_a_pilot_y_ft_sec2)) +
        #                         0.8*math.fabs(sim.get_property_value(c.accelerations_a_pilot_z_ft_sec2))))

        # HeadingControlTask.mostRecentRewards['heading_r'] = heading_r
        # HeadingControlTask.mostRecentRewards['alt_r'] = alt_r
        # HeadingControlTask.mostRecentRewards['angle_speed_r'] = angle_speed_r
        # HeadingControlTask.mostRecentRewards['reward'] = reward
        # reward = 0.4*heading_r + 0.4*alt_r + 0.2*angle_speed_r

        # Heading reward:
        worstCaseHeadingDiff = 90.0
        headDelt = sim.get_property_value(c.delta_heading)
        heading_reward = (1 - (abs(headDelt) / worstCaseHeadingDiff)) / 2.0

        # Altitude reward/penalty:
        worstCaseAltitudeScore = 300.0
        altDelt = sim.get_property_value(c.delta_altitude)

        if altDelt < worstCaseAltitudeScore:
           altitude_reward = (1 - (abs(altDelt) / worstCaseAltitudeScore)) / 2.0
        else:
           # For every 1000 feet beyond 300, lose another point:
           altitude_reward = -((altDelt - worstCaseAltitudeScore) / 1000.0)

        targetChange_reward = self.numTargetChanges

        # Reward goes up the longer you stay in the environment, hopefully
        # encouraging getting on heading and staying at appropriate altitude.
        reward = targetChange_reward + heading_reward + altitude_reward

        self.mostRecentRewards = {
         'heading_reward': heading_reward,
         'altitude_reward': altitude_reward,
         'targetChange_reward': targetChange_reward,
         'reward': reward}

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward

    def get_reward_maintain_altitude(self, state, sim):
        """For sanity's sake, see if we can train an agent just to maintain
        altitude."""
        d_alt = abs(sim.get_property_value(c.delta_altitude))

        worstCaseAltitudeScore = 1000
        worstCaseAltitudePenalty = 2000

        # reward = 0
        # if d_alt <= worstCaseAltitudeScore:  # Gain points for being within altitude.
        #    reward = (worstCaseAltitudeScore - d_alt) / worstCaseAltitudeScore
        # else:  # Lose points for being outside altitude.
        #    reward = -((d_alt - worstCaseAltitudeScore) / (worstCaseAltitudePenalty - worstCaseAltitudeScore))

        # Maybe make sure there's no losing score? Staying in the game is always
        # best:
        reward = (worstCaseAltitudePenalty - d_alt) / worstCaseAltitudePenalty

        # If the sim is about to stop due to being out of the acceptable
        # altitude range, take a big hit:
        if d_alt >= 2000.0:
            reward = -100.0

        # If you managed to last until the end of the scenario without going
        # outside the acceptable altitude, get a big bonus:
        if sim.get_property_value(c.simulation_sim_time_sec) >= 2000.0:
            reward = 100.0

        self.mostRecentRewards = {
         'delta_alt': d_alt,
         'reward': reward,
        }

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward

    def get_reward_maintain_altitude_and_heading(self, state, sim):
        """Train on maintaining altitude AND heading."""
        d_alt = abs(sim.get_property_value(c.delta_altitude))

        worstCaseAltitudeScore = 1000
        worstCaseAltitudePenalty = 2000
        return false

    def is_terminal_maintain_altitude(self, state, sim):
        # Run for a maximum of 2000 seconds or until we're way outside the
        # the altitude requirements, or put the plane in a bad state.
        retVal = sim.get_property_value(c.simulation_sim_time_sec)>=2000 or \
                 math.fabs(sim.get_property_value(c.delta_altitude)) >= self.worstCaseAltitudeDelta or \
                 bool(sim.get_property_value(c.detect_extreme_state))

        if retVal:
           self.simStopInfo = f"Time (sec): {sim.get_property_value(c.simulation_sim_time_sec)}. Delta alt: {sim.get_property_value(c.delta_altitude)}. Extreme state: {sim.get_property_value(c.detect_extreme_state)}"

        return retVal


    def is_terminal(self, state, sim):

        return self.is_terminal_maintain_altitude(state, sim)

        # if accelerations are too high stop the simulation
        acc = 36 # 1.2G
        if (sim.get_property_value(c.simulation_sim_time_sec)>10):
            if math.fabs(sim.get_property_value(c.accelerations_a_pilot_x_ft_sec2)) > acc or math.fabs(sim.get_property_value(c.accelerations_a_pilot_y_ft_sec2)) > acc or math.fabs(sim.get_property_value(c.accelerations_a_pilot_z_ft_sec2)) > acc:
                self.stopReason = f"IGNORING Acceleration too high. x: {sim.get_property_value(c.accelerations_a_pilot_x_ft_sec2)}, y: {sim.get_property_value(c.accelerations_a_pilot_y_ft_sec2)}, z: {sim.get_property_value(c.accelerations_a_pilot_z_ft_sec2)}"
                #return True

        # Change heading every 150 seconds
        ############################
        #  time > steady flight? How is that changing heading every 150 seconds?
        ############################
        if sim.get_property_value(c.simulation_sim_time_sec) >= sim.get_property_value(c.steady_flight):
            # if the traget heading was not reach before, we stop the simulation
            if math.fabs(sim.get_property_value(c.delta_heading)) > 10:
                self.stopReason = f"Delta heading too high: {sim.get_property_value(c.delta_heading)} > 10"
                return True

            # We set the new target heading every 150s in an incremental difficulty: 10, -20, 30, -40, 50, -60, 70, -80, 90
            new_alt = sim.get_property_value(c.target_altitude_ft)
            angle = int(sim.get_property_value(c.steady_flight)/150) * 10
            if int(sim.get_property_value(c.steady_flight)/150) % 2 == 1:
                new_heading = sim.get_property_value(c.target_heading_deg) + angle
            else:
                new_heading = sim.get_property_value(c.target_heading_deg) - angle

            new_heading = (new_heading +360) % 360

            self.otherInfo = f'Change # {self.numTargetChanges}: {sim.get_property_value(c.simulation_sim_time_sec)} (Altitude: {sim.get_property_value(c.target_altitude_ft)} -> {new_alt}, Heading: {sim.get_property_value(c.target_heading_deg)} -> {new_heading})'
            self.numTargetChanges += 1

            sim.set_property_value(c.target_altitude_ft, new_alt)
            sim.set_property_value(c.target_heading_deg, new_heading)

            sim.set_property_value(c.steady_flight,sim.get_property_value(c.steady_flight)+150)

            # End up the simulation after 1500 secondes or if the aircraft is under or above 300 feet of its target altitude or velocity under 400f/s

        retVal = sim.get_property_value(c.simulation_sim_time_sec)>=2000 or math.fabs(sim.get_property_value(c.delta_altitude)) >= self.worstCaseAltitudeDelta or bool(sim.get_property_value(c.detect_extreme_state))
        if retVal:
            self.simStopInfo = f"Time (sec): {sim.get_property_value(c.simulation_sim_time_sec)}. Delta alt: {sim.get_property_value(c.delta_altitude)}. Extreme state: {sim.get_property_value(c.detect_extreme_state)}"

        return retVal


    def render(self, mode='human'):
        # Output everything, then reset all so the outputs aren't duplicated:
        outString = "Rewards: "
        for key,value in self.mostRecentRewards.items():
           outString += f"{key}: {round(value, 6)} "
        if self.stopReason is not None:
           outString += f'\n\tStop: "{self.stopReason}" '
        if self.otherInfo is not None:
           outString += f'\n\tOther: "{self.otherInfo}" '
        if self.simStopInfo is not None:
           outString += f'\n\tInfo: "{self.simStopInfo}" '

        outString += f'\n\tSteps: {self.stepCount}. Sim seconds: {self.simTime}'

        print(outString)

        self.mostRecentRewards = {}
        self.stopReason = None
        self.otherInfo = None
        self.simStopInfo = None
