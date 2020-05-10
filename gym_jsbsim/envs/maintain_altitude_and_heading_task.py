from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
A task that just asks the agent to maintain its starting altitude and heading
indefinitely.

@author: Joseph Williams

Possible enhancements:
  - Get this task up-to-date with task.py changes. It is so far out of date,
    it would no longer function.  
"""

class MaintainAltitudeAndHeadingTask(Task):

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
       self.worstCaseAltitudeDelta = 3000
       self.worstCaseHeadingDelta = 110

       # Longer sim time to hopefully force the agent to not just 'get lucky':
       self.maxSimTime = 8000

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in MaintainAltitudeAndHeadingTask.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       print(f"Prop min/maxes:\n{self.observation_minMaxes}")
       # The deltaAltitude is 40k.  Since we'll limit our aircraft to a
       # delta-altitude of 5k, change that variable:
       self.observation_minMaxes[0] = [-self.worstCaseAltitudeDelta,
                                       self.worstCaseAltitudeDelta]

       # The deltaHeading will get stopped at 110 degrees off, but let the
       # [-180, 180] range remain in place.

       # Assume floating point:
       # All actions are [-1, 1] except throttle which goes [0, 0.9]:
       fullActionSpace = spaces.Box(low=np.array([-1.0, -1.0, 0, -1.0]),
                                      high=np.array([1.0, 1.0, 0.9, 1.0]),
                                      dtype=np.float32)

       # Our action space if we don't let them control the rudder:
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
        """Reward a plane for staying on altitude and heading."""
        d_alt = abs(sim.get_property_value(c.delta_altitude))
        altitudeReward = (self.worstCaseAltitudeDelta - d_alt) / self.worstCaseAltitudeDelta

        d_heading = abs(sim.get_property_value(c.delta_heading))
        headingReward = (self.worstCaseHeadingDelta - d_heading) / self.worstCaseHeadingDelta

        reward = (0.5 * altitudeReward) + (0.5 * headingReward)

        # If you managed to last until the end of the scenario without going
        # outside the acceptable altitude or heading, get a big bonus:
        if sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime:
            reward = 100.0

        self.mostRecentRewards = {
         'delta_alt': d_alt,
         'delta_heading': d_heading,
         'reward': reward,
        }

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward

    def is_terminal(self, state, sim):
        # Run for a maximum of 2000 seconds or until we're way outside the
        # the altitude requirements, or put the plane in a bad state.
        retVal = sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime or \
                 math.fabs(sim.get_property_value(c.delta_altitude)) >= self.worstCaseAltitudeDelta or \
                 math.fabs(sim.get_property_value(c.delta_heading)) >= self.worstCaseHeadingDelta or \
                 bool(sim.get_property_value(c.detect_extreme_state))

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
