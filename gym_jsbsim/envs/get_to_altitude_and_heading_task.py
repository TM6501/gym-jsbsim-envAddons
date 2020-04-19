from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

class GetToAltitudeAndHeadingTask(Task):

    def getStartHeadingAndAltitude(self):
       # 4 choices of headings: Big/small left/right turns.
       # 5 choices for altitudes: 4 up/down and one random. The random is added
       #                          to make sure that the heading and altitude
       #                          changes don't line up; we don't want the
       #                          agent associating turn directions and
       #                          altitude changes.

       # It is possible this function is called before __init__. If
       # initHeadingChoice and initAltitudeChoice haven't yet been defined,
       # define them here:

       try:
           self.initHeadingChoice += 1
       except:
           self.initHeadingChoice = random.randint(0,3)

       try:
           self.initAltitudeChoice += 1
       except:
           self.initAltitudeChoice = random.randint(0,4)

       if self.initHeadingChoice > 3:
           self.initHeadingChoice = 0

       if self.initAltitudeChoice > 4:
           self.initAltitudeChoice = 0

       heading = 100.0
       if self.initHeadingChoice == 0:
           heading = random.uniform(25.0, 35.0)
       elif self.initHeadingChoice == 1:
           heading = random.uniform(55.0, 65.0)
       elif self.initHeadingChoice == 2:
           heading = random.uniform(135.0, 145.0)
       elif self.initHeadingChoice == 3:
           heading = random.uniform(165.0, 175.0)

       altitude = 10000.0
       if self.initAltitudeChoice == 0:
           altitude = random.uniform(12500.0, 13500.0)
       elif self.initAltitudeChoice == 1:
           altitude = random.uniform(11000.0, 12000.0)
       elif self.initAltitudeChoice == 2:
           altitude = random.uniform(8000.0, 9000.0)
       elif self.initAltitudeChoice == 3:
           altitude = random.uniform(6500.0, 7500.0)
       else:
           tempInt = random.randint(0,1)
           if tempInt == 0:
               altitude = random.uniform(6500.0, 9000.0)
           else:
               altitude = random.uniform(11000.0, 13500.0)

       # DEBUG: Try smaller diffs from start:
       # Above larger heading/altitue delta-targets are not allowing the
       # agent to train.
       # if random.randint(0,1) == 0:
       #     heading = random.uniform(65.0, 75.0)
       # else:
       #     heading = random.uniform(125.0, 135.0)
       #
       # if random.randint(0,1) == 0:
       #     altitude = random.uniform(8500.0, 9000.0)
       # else:
       #     altitude = random.uniform(11000.0, 11500.0)
       # END DEBUG

       return heading, altitude

    def get_initial_conditions(self):
       # Chang the start heading and altitude:
       initHeading, initAltitude = self.getStartHeadingAndAltitude()
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
         c.target_altitude_ft: initAltitude,
         c.fcs_throttle_cmd_norm: 0.8,
         c.fcs_mixture_cmd_norm: 1,
         c.gear_gear_pos_norm : 0,
         c.gear_gear_cmd_norm: 0,
         c.steady_flight:150}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # How far can we get off of our goal before ending the scenario?
       self.worstCaseAltitudeDelta = 2000 # 7500
       self.worstCaseHeadingDelta = 20 # 180

       # How far can we get off of our goal before we start getting negative rewards?
       self.zeroSwapAltitudeDelta = 4500
       self.zeroSwapHeadingDelta = 110

       # Random is too random... we need to force the agent to see left and
       # right turns and only be successful when it accomplishes both.
       # initHeadingChoice may have already been defined:
       try:
           self.initHeadingChoice
       except:
           self.initHeadingChoice = random.randint(0,3)

       try:
           self.initAltitudeChoice
       except:
           self.initAltitudeChoice = random.randint(0,4)

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in self.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

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

       # self.action_space = fullActionSpace
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

    def is_terminal(self, state, sim):
        timeOut = sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime
        altitudeOut = math.fabs(sim.get_property_value(c.delta_altitude)) >= 15000
        extremeOut = bool(sim.get_property_value(c.detect_extreme_state))

        retVal = timeOut or altitudeOut or extremeOut
        return retVal

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
        # reward = self.get_staggered_reward_altitude_and_heading_with_negatives(
        #   state, sim, numPositiveAltitudeStaggerLevels=25,
        #   numPositiveHeadingStaggerLevels=25, altitudeWorth=0.5,
        #   headingWorth=0.5, numNegativeAltitudeStaggerLevels=25,
        #   numNegativeHeadingStaggerLevels=25, penaltyMultiplier=0.1)
        reward = self.get_staggered_reward_altitude_and_heading(state, sim,
          numAltitudeStaggerLevels=10, numHeadingStaggerLevels=10,
          headingWorth=0.5, altitudeWorth=0.5)

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        return reward
