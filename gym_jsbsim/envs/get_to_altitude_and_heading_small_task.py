from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

class GetToAltitudeAndHeadingSmallTask(Task):

    def getStartHeadingAndAltitudeChanges(self):
       # Our goal is to make a very small, but precise and quick adjustment
       # to our altitude and/or heading. Choose said targets here:

       # Choose our altitude change:
       altitudeChoice = random.randint(-1, 1)

       altitude = 0
       if altitudeChoice == -1:
           altitude = random.uniform(-800.0, -1200.0)
       elif altitudeChoice == 0:
           altitude = random.uniform(-200.0, 200.0)
       else:
           altitue = random.uniform(800.0, 1200.0)

       # Choose our heading change:
       headingChoice = random.randint(-1, 1)

       heading = 0
       if headingChoice == -1:
           heading = random.uniform(-8.0, -12.0)
       elif headingChoice == 0:
           heading = random.uniform(-4.0, 4.0)
       else:
           heading = random.uniform(8.0, 12.0)

       return heading, altitude

    def get_initial_conditions(self):
       # Chang the start heading and altitude:
       headingDiff, altitudeDiff = self.getStartHeadingAndAltitudeChanges()

       headingGoal = 100 + headingDiff
       altitudeGoal = 10000 + altitudeDiff

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
         c.target_heading_deg: headingGoal,
         c.target_altitude_ft: altitudeGoal,
         c.fcs_throttle_cmd_norm: 0.8,
         c.fcs_mixture_cmd_norm: 1,
         c.gear_gear_pos_norm : 0,
         c.gear_gear_cmd_norm: 0,
         c.steady_flight:150}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # How far can we get off of our goal before ending the scenario. We need
       # to set tiny ranges to force precision from our agent:
       self.worstCaseAltitudeDelta = 2000
       self.worstCaseHeadingDelta = 20

       # How far can we get off of our goal before we start getting negative rewards?
       # For now we're not using these:
       # self.zeroSwapAltitudeDelta = 4500
       # self.zeroSwapHeadingDelta = 110

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

       self.action_space = zeroRudderActionSpace

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
       retObs = np.array(observation).reshape(-1)

       for i in range(len(retObs)):
          retObs[i] = (((retObs[i] - self.observation_minMaxes[i][0]) / (self.observation_minMaxes[i][1] - self.observation_minMaxes[i][0])) - 0.5) * 2.0

          # Make sure we stay within [-1, 1]:
          retObs[i] = min(1.0, max(-1.0, retObs[i]))

       self.renderVariables['observation'] = self.detailedObservationOutput(observation, retObs)
       return retObs

    def get_reward(self, state, sim):
        """Reward a plane for staying on altitude and heading."""
        reward = self.get_staggered_reward_altitude_and_heading(
          state, sim, numAltitudeStaggerLevels=10, numHeadingStaggerLevels=10,
          altitudeWorth=0.5, headingWorth=0.5)

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward
