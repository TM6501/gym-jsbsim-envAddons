from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
@author: Joseph Williams

This task is one of many increasing steps of difficulty as the agent-training
abilities were tested and developed. It asks the agent to get to a heading
while maintaining its current altitude.

Possible enhancements:
  - Ensure it is up-to-date with current task.py changes.
  - Add infinite-fuel change.
"""

class MaintainAltitudeAndGetToHeadingTask(Task):

    def getStartHeading(self):
        # 4 choices of ranges: Big/small left/right turns:
        # It is possible this function is called before __init__. If
        # initHeadingChoice hasn't been defined, define it here:
        try:
            self.initHeadingChoice += 1
        except:
            self.initHeadingChoice = random.randint(0,3)

        if self.initHeadingChoice > 3:
            self.initHeadingChoice = 0

        heading = 100.0
        if self.initHeadingChoice == 0:
            heading = random.uniform(25.0, 35.0)
        elif self.initHeadingChoice == 1:
            heading = random.uniform(55.0, 65.0)
        elif self.initHeadingChoice == 2:
            heading = random.uniform(135.0, 145.0)
        elif self.initHeadingChoice == 3:
            heading = random.uniform(165.0, 175.0)

        return heading

    # With a change-heading task, initial conditions need to be somewhat
    # randomized.  Provide that functionality:
    def get_initial_conditions(self):
       # Change the heading we start at:
       initHeading = int(self.getStartHeading())
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
         c.target_heading_deg: initHeading,
         c.target_altitude_ft: 10000,
         c.fcs_throttle_cmd_norm: 0.8,
         c.fcs_mixture_cmd_norm: 1,
         c.gear_gear_pos_norm : 0,
         c.gear_gear_cmd_norm: 0,
         c.steady_flight:150}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # How screwed is too screwed?
       self.worstCaseAltitudeDelta = 3000
       self.worstCaseHeadingDelta = 110

       # Random is too random... we need to force the agent to see left and
       # right turns and only be successful when it accomplishes both.
       # initHeadingChoice may have already been defined:
       try:
           self.initHeadingChoice
       except:
           self.initHeadingChoice = random.randint(0,3)

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in self.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       # The deltaAltitude is 40k.  Change the min/max of the altitude
       # difference to match that:
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

       # self.action_space = zeroRudderActionSpace
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
        reward = self.get_staggered_reward_altitude_and_heading(state, sim)

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward
