from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
A task that asks the agent to maintain its starting heading while moving to
a target altitude.

@author: Joseph Williams

Possible enhancements:
  - Get this task up-to-date with task.py changes. It is so far out of date,
    it would no longer function.
  - Infinite fuel option.
"""

class MaintainHeadingAndGetToAltitudeTask(Task):

    def getStartAltitude(self):
        # 4 choices: Big/small up/down:
        # Since this function can be called before __init__, we need to check
        # that initAltitudeChoice is defined:
        try:
            self.initAltitudeChoice += 1
        except:
            self.initAltitudeChoice = random.randint(0,3)

        if self.initAltitudeChoice > 3:
            self.initAltitudeChoice = 0

        altitude = 10000
        if self.initAltitudeChoice == 0:
            altitude = random.uniform(12500.0, 13500.0)
        elif self.initAltitudeChoice == 1:
            altitude = random.uniform(11000.0, 12000.0)
        elif self.initAltitudeChoice == 2:
            altitude = random.uniform(8000.0, 9000.0)
        else:
            altitude = random.uniform(6500.0, 7500.0)

        return altitude

    # With a change-heading task, initial conditions need to be somewhat
    # randomized.  Provide that functionality:
    def get_initial_conditions(self):
       # Change the altitude we aim for:
       initAltitude = self.getStartAltitude()
       self.init_conditions = {
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
         c.target_altitude_ft: initAltitude,
         c.fcs_throttle_cmd_norm: 0.8,
         c.fcs_mixture_cmd_norm: 1,
         c.gear_gear_pos_norm : 0,
         c.gear_gear_cmd_norm: 0,
         c.steady_flight:150}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # How screwed is too screwed?
       self.worstCaseAltitudeDelta = 7500
       self.worstCaseHeadingDelta = 110

       # Random is too random... we need to force the agent to see left and
       # right turns and only be successful when it accomplishes both.
       # initHeadingChoice may have already been defined:
       try:
           self.initAltitudeChoice
       except:
           self.initAltitudeChoice = random.randint(0,3)

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in self.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       # The deltaAltitude is 40k.  Since we'll limit our aircraft to a
       # delta-altitude of 5k, change that variable:
       self.observation_minMaxes[0] = [-self.worstCaseAltitudeDelta,
                                       self.worstCaseAltitudeDelta]

       # Assume floating point:
       # All actions are [-1, 1] except throttle which goes [0, 0.9]:
       fullActionSpace = spaces.Box(low=np.array([-1.0, -1.0, 0, -1.0]),
                                      high=np.array([1.0, 1.0, 0.9, 1.0]),
                                      dtype=np.float32)

       # Our action space if we don't let them control the rudder:
       zeroRudderActionSpace = spaces.Box(low=np.array([-1.0, -1.0, 0]),
                                      high=np.array([1.0, 1.0, 0.9]),
                                      dtype=np.float32)

       self.action_space = fullActionSpace

       # Bangbang controls have different input requirements:
       if not self.floatingAction:
          self.action_space = spaces.Discrete(18)

       self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(9,),
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

       self.renderVariables['observation'] = self.detailedObservationOutput(observation, retObs)
       return retObs

    def get_reward(self, state, sim):
        # reward = self.get_simple_reward(state, sim)
        reward = self.get_staggered_reward_altitude_and_heading(state, sim, numAltitudeStaggerLevels=50, numHeadingStaggerLevels=50)

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward
