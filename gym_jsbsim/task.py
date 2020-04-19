from types import MethodType
import numpy as np
import math
import gym
from gym.spaces import Box, Discrete
from gym_jsbsim.catalogs.catalog import Catalog as c

class Task:
    """

        A class to subclass in order to create a task with its own observation variables,

        action variables, termination conditions and agent_reward function.

    """
    jsbsim_freq = 60  # Sim moves as 60Hz
    agent_interaction_steps = 30  # We get asked for input every 30 frames. (0.5 seconds)
    aircraft_name = 'A320' # 'f16'

    terminalReasons = [0, 0, 0, 0]

    def __init__(self):

        # Get our state and action vars:
        self.action_var = self.get_action_var()
        self.state_var = self.get_state_var()
        self.init_conditions = self.get_initial_conditions()

        # set default output to state_var
        #if self.output is None:
        self.output = self.state_var

        # Variables we output on render:
        self.renderVariables = {}

        # Debug to count:
        self.stepCount = 0
        self.simTime = 0

        # How far off do we let the agent get before we kill the scenario and
        # start over?
        self.worstCaseAltitudeDelta = 4000
        self.worstCaseHeadingDelta = 180

        # How far from our goal do we let our agent get before it switches from
        # positive to negative rewards:
        self.zeroSwapAltitudeDelta = 3000
        self.zeroSwapHeadingDelta = 110

        # Longer sim time to hopefully force the agent to not just 'get lucky':
        self.maxSimTime = 8000

        # modify Catalog to have only the current task properties
        names_away = []
        for name,prop in c.items():
            if not( prop in self.action_var or prop in self.state_var or prop in self.init_conditions or prop in self.output) :
                names_away.append(name)
        for name in names_away:
            c.pop(name)

    def get_simple_reward_altitude_and_heading(self, state, sim):
        """Reward a plane for staying on altitude and heading."""
        d_alt = abs(sim.get_property_value(c.delta_altitude))
        altitudeReward = (self.worstCaseAltitudeDelta - d_alt) / self.worstCaseAltitudeDelta

        d_heading = abs(sim.get_property_value(c.delta_heading))
        headingReward = (self.worstCaseHeadingDelta - d_heading) / self.worstCaseHeadingDelta

        reward = (0.5 * altitudeReward) + (0.5 * headingReward)

        self.renderVariables['rewards'] = {
          'delta_alt': d_alt,
          'delta_heading': d_heading,
          'reward': reward,
        }

        return reward

    def get_even_ranges(self, maxValue, numberSteps):
        """Get a set of even ranges to stagger increasing/decreasing rewards.
        """
        stepSize = maxValue / numberSteps
        current = stepSize
        retList = []
        for i in range(numberSteps):
            retList.append(round(current, 2))
            current += stepSize
        return retList

    def get_staggered_reward_altitude_and_heading_with_negatives(self, state,
      sim, numPositiveAltitudeStaggerLevels=10, numPositiveHeadingStaggerLevels=10,
      numNegativeAltitudeStaggerLevels=10, numNegativeHeadingStaggerLevels=10,
      penaltyMultiplier=0.1, altitudeWorth=0.5, headingWorth=0.5):
        """Return a reward with staggered changes, except instead of those
        positive rewards stopping at worstCaseXDelta, they stop at
        zeroSwapXDelta and the range from zeroSwapXDelta to worstCaseXDelta is
        treated as penalty instead of reward."""

        d_alt = abs(sim.get_property_value(c.delta_altitude))
        d_heading = abs(sim.get_property_value(c.delta_heading))
        altitudeReward = 0
        headingReward = 0

        if d_alt <= self.zeroSwapAltitudeDelta:
            altitudeMaxes = self.get_even_ranges(self.zeroSwapAltitudeDelta, numPositiveAltitudeStaggerLevels)
            # 1/X per value:
            valuePerAltitude = (1.0 / float(numPositiveAltitudeStaggerLevels)) * altitudeWorth

            for altMax in altitudeMaxes:
                if d_alt <= altMax:
                    altitudeReward += (altMax - d_alt) / altMax
            altitudeReward *= valuePerAltitude

        else:  # Delta altitude is beyond where we get a reward:
            # Get a set of ranges from 0 to diff(zeroSwap, worstCase):
            altitudeMaxes = self.get_even_ranges(
              self.worstCaseAltitudeDelta - self.zeroSwapAltitudeDelta,
              numNegativeAltitudeStaggerLevels)

            valuePerAltitude = (1.0 / float(numNegativeAltitudeStaggerLevels)) * altitudeWorth

            # Flip it so we're measuring distance from worst case:
            d_alt = self.worstCaseAltitudeDelta - d_alt

            for altMax in altitudeMaxes:
                if d_alt <= altMax:
                    altitudeReward += (altMax - d_alt) / altMax

            # Scale it and negate it:
            altitudeReward *= valuePerAltitude
            altitudeReward *= penaltyMultiplier
            altitudeReward *= -1.0

        # Repeat the process for heading:
        if d_heading <= self.zeroSwapHeadingDelta:
            headingMaxes = self.get_even_ranges(self.zeroSwapHeadingDelta, numPositiveHeadingStaggerLevels)
            # 1/X per value:
            valuePerHeading = (1.0 / float(numPositiveHeadingStaggerLevels)) * headingWorth

            for headingMax in headingMaxes:
                if d_heading <= headingMax:
                    headingReward += (headingMax - d_heading) / headingMax
            headingReward *= valuePerHeading

        else:  # Delta heading is beyond where we get a reward
            headingMaxes = self.get_even_ranges(
              self.worstCaseHeadingDelta - self.zeroSwapHeadingDelta,
              numNegativeHeadingStaggerLevels)

            valuePerHeading = (1.0 / float(numNegativeHeadingStaggerLevels)) * headingWorth

            # Flip it so we're measuring distance from worst case:
            d_heading = self.worstCaseHeadingDelta - d_heading

            for headingMax in headingMaxes:
                if d_heading <= headingMax:
                    headingReward += (headingMax - d_heading) / headingMax

            # Scale and negate:
            headingReward *= valuePerHeading
            headingReward *= penaltyMultiplier
            headingReward *= -1.0

        reward = altitudeReward + headingReward

        # With the render variables possibly being used to create CSV files,
        # we need to spread things out to one per line, even if that makes it
        # hard for humans to read.  For now, we'll also forgo the detailed
        # reward vectors:
        self.renderVariables['rewards'] = {
          'altitude': sim.get_property_value(c.position_h_sl_ft),
          'targetAltitude': sim.get_property_value(c.target_altitude_ft),
          'altitudeReward': altitudeReward,
          'heading': sim.get_property_value(c.attitude_psi_deg),
          'targetHeading': sim.get_property_value(c.target_heading_deg),
          'headingReward': headingReward,
          'totalReward': reward,
        }

        return reward

    def get_staggered_reward_altitude_and_heading(self, state, sim,
      numAltitudeStaggerLevels=10, numHeadingStaggerLevels=10,
      altitudeWorth=0.5, headingWorth=0.5):
        """Return a reward where each section of the goal is more important than
        the section before it. Hopefully, this will encourage the agent to be
        more precise as the difference between 500 feet off and 0 feet off
        becomes more critical than the difference 2500 and 2000."""
        altitudeMaxes = self.get_even_ranges(self.worstCaseAltitudeDelta, numAltitudeStaggerLevels)
        headingMaxes = self.get_even_ranges(self.worstCaseHeadingDelta, numHeadingStaggerLevels)

        # 1/X per value:
        valuePerAltitude = (1.0 / float(numAltitudeStaggerLevels)) * altitudeWorth
        valuePerHeading = (1.0 / float(numHeadingStaggerLevels)) * headingWorth

        # Get the altitude full reward:
        d_alt = abs(sim.get_property_value(c.delta_altitude))
        altitudeReward = 0
        # For debugging, record all altitude sub-rewards
        altitudeRewards = []
        for altMax in altitudeMaxes:
            if d_alt <= altMax:
                altitudeRewards.append( (altMax - d_alt) / altMax )
                altitudeReward += (altMax - d_alt) / altMax

        altitudeReward *= valuePerAltitude

        d_heading = abs(sim.get_property_value(c.delta_heading))
        headingReward = 0
        # For debugging, record all heading sub-rewards
        headingRewards = []
        for headingMax in headingMaxes:
            if d_heading <= headingMax:
                headingRewards.append( (headingMax - d_heading) / headingMax )
                headingReward += (headingMax - d_heading) / headingMax

        headingReward *= valuePerHeading

        reward = altitudeReward + headingReward

        # With the render variables possibly being used to create CSV files,
        # we need to spread things out to one per line, even if that makes it
        # hard for humans to read.  For now, we'll also forgo the detailed
        # reward vectors:
        self.renderVariables['rewards'] = {
          'altitude': sim.get_property_value(c.position_h_sl_ft),
          'targetAltitude': sim.get_property_value(c.target_altitude_ft),
          'altitudeReward': altitudeReward,
          'heading': sim.get_property_value(c.attitude_psi_deg),
          'targetHeading': sim.get_property_value(c.target_heading_deg),
          'headingReward': headingReward,
          'totalReward': reward,
        }

        return reward

    def get_reward(self, state, sim):
        reward = self.get_simple_reward_altitude_and_heading

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward

    def is_terminal(self, state, sim):
        timeOut = sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime
        altitudeOut = math.fabs(sim.get_property_value(c.delta_altitude)) >= self.worstCaseAltitudeDelta
        headingOut = math.fabs(sim.get_property_value(c.delta_heading)) >= self.worstCaseHeadingDelta
        extremeOut = bool(sim.get_property_value(c.detect_extreme_state))

        retVal = timeOut or altitudeOut or headingOut or extremeOut

        # Count number of times each failure occurs:
        outputFailureCountsMod = 0  # Set to anything but 0 to count:
        if retVal and outputFailureCountsMod != 0:
            if timeOut:
                Task.terminalReasons[0] += 1
            if altitudeOut:
                Task.terminalReasons[1] += 1
            if headingOut:
                Task.terminalReasons[2] += 1
            if extremeOut:
                Task.terminalReasons[3] += 1

            if sum(Task.terminalReasons) % outputFailureCountsMod == 0:
                print(f"""Terminals - Time: {Task.terminalReasons[0]}, \
Altitude: {Task.terminalReasons[1]}, Heading: {Task.terminalReasons[2]}, \
Extreme: {Task.terminalReasons[3]}""")

        return retVal

    def get_observation_var(self):
        return self.state_var

    def get_initial_conditions(self):
        # Define some default initial conditions:
        return {
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

    def get_action_var(self):
        # Define our default action var, but allow sub-tasks to override:
        return [
          c.fcs_aileron_cmd_norm,
          c.fcs_elevator_cmd_norm,
          c.fcs_throttle_cmd_norm,
          c.fcs_rudder_cmd_norm,
        ]

    def get_state_var(self):
        # All environments so far share a state var. Let them override if
        # they need to, though:
        return [
          c.delta_altitude,        # Target Alt - Alt above MSL, ft [-40000, 40000].  Normalize based on maximum allowed by this task.
          c.delta_heading,         # Heading Diff, degrees reduced to range [-180, 180]
          c.velocities_v_down_fps, # Velocity downward, ft/s [-2200, 2200]
          c.velocities_vc_fps,     # Airspeed, knots [0, 4400]
          c.velocities_p_rad_sec,  # Roll rate, rad / s  [-2 * pi, 2 * pi]
          c.velocities_q_rad_sec,  # Pitch rate, rad / s [-2 * pi, 2 * pi]
          c.velocities_r_rad_sec,  # Yaw rate, rad / s [-2 * pi, 2 * pi]
          # Newly added:
          c.attitude_pitch_rad,    # Pitch, rad [-0.5 * pi, 0.5 * pi]
          c.attitude_roll_rad,     # Roll, rad [-pi, pi]
        ]

    def get_output(self):
        return self.output

    def get_observation_space(self):
        """
        Get the task's observation Space object

        :return : spaces.Tuple composed by spaces of each property.
        """

        space_tuple = ()

        for prop in self.state_var:
            if prop.spaces is Box:
                space_tuple += (Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype='float'),)
            elif prop.spaces is Discrete:
                space_tuple += (Discrete(prop.max - prop.min + 1),)
        return gym.spaces.Tuple(space_tuple)

    def get_action_space(self):
        """
        Get the task's action Space object

        :return : spaces.Tuple composed by spaces of each property.
        """
        space_tuple = ()

        for prop in self.action_var:
            if prop.spaces is Box:
                space_tuple += (Box(low=np.array([prop.min]), high=np.array([prop.max]), dtype='float'),)
            elif prop.spaces is Discrete:
                space_tuple += (Discrete(prop.max - prop.min + 1),)
        return gym.spaces.Tuple(space_tuple)

    def render(self, mode='human', **kwargs):
        # Output everything, then reset all so the outputs aren't duplicated:
        outString = f"SimTime: {round(self.simTime, 4)}\nSimSteps: {self.stepCount}"
        # For the rest of the values, make sure we have the same order every
        # time by using the sorted list of keys:
        for key in sorted(self.renderVariables.keys()):
           value = self.renderVariables[key]
           if isinstance(value, dict):
               for key2 in sorted(value.keys()):
                   value2 = value[key2]
                   outString += f"\n{key2}: {round(value2, 4)} "
           else:
               outString += f"\n{key}: {round(value, 4)}"

        print(outString)
        self.renderVariables = {}

    def reset(self):
       # Variables we want to track and output at render time. Add agent
       # selctions so that each render call looks the same:
       self.renderVariables = {
         'action': {
           'aileron': 0,
           'elevator': 0,
           'throttle': 0,
           'rudder': 0
         }
       }

       # Debug to count:
       self.stepCount = 0
       self.simTime = 0

    def notifyAction(self, action):
        """Record the action the agent took for rendering."""
        names = ['aileron', 'elevator', 'throttle', 'rudder']

        actionDict = {}
        for i in range(min(len(names), len(action))):
            actionDict[names[i]] = round(action[i], 4)

        self.renderVariables['action'] = actionDict

    def detailedObservationOutput(self, startObs, endObs):
        names = ['delta_altitude', # Target Alt - Alt above MSL, ft [-40000, 40000].  Normalize based on maximum allowed by this task.
                 'delta_heading',  # Heading Diff, degrees reduced to range [-180, 180]
                 'velocities_v_down_fps', # Velocity downward, ft/s [-2200, 2200]
                 'velocities_vc_fps',     # Airspeed, knots [0, 4400]
                 'velocities_p_deg_sec',  # Roll rate, rad / s  [-2 * pi, 2 * pi]
                 'velocities_q_deg_sec',  # Pitch rate, rad / s [-2 * pi, 2 * pi]
                 'velocities_r_deg_sec',  # Yaw rate, rad / s [-2 * pi, 2 * pi]
                 'attitude_pitch_deg',    # Pitch, rad [-0.5 * pi, 0.5 * pi]
                 'attitude_roll_deg']

        # Convert all of the radians to degrees to make the output easier to read:
        convertedObservation = []
        for i in range(9):
           if i >= 4:
              convertedObservation.append(math.degrees(startObs[i][0]))
           else:
              convertedObservation.append(startObs[i][0])

        # Create a dictionary to give the input obs and output obs. Separate
        # the values from their normalized values to make CSV output easier:
        retDict = {}
        for i in range(9):
            retDict[names[i]] = round(convertedObservation[i], 2)
            retDict[names[i] + '_normalized'] = round(endObs[i], 4)

        return retDict

    def convertObservation(self, observation):
        return observation

    def get_jsbsim_freq(self):
        return self.jsbsim_freq

    def get_agent_interaction_steps(self):
        return self.agent_interaction_steps
