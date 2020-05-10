from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
@author: Joseph Williams

This task asks the agent to make a very small altitude and heading change.
The directions of each change is randomized.

Possible enhancements:
  - Determine why the observation_minMaxes changes for altitude seem mandatory
    but for heading seem optional. The agent trains to a high level of skill
    without it. Does this mean we could have accomplished the full-task
    training by just weighting the altitude goal higher than heading? Or by
    adding many more altitude reward stagger levels? Worth exploring.
"""

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
       # Change the start heading and altitude:
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
         c.steady_flight:150,
         # Start with full fuel and constantly refuel it. Agents trained in this
         # manner will not deal well with scenarios with significantly less
         # fuel. The weight of the aircraft is an import factor.
         c.propulsion_tank0_contents_lbs: 24000.0,
         c.propulsion_tank1_contents_lbs: 24000.0,
         c.propulsion_refuel: 1}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # How far can we get off of our goal before ending the scenario. We need
       # to set tiny ranges to force precision from our agent:
       self.worstCaseAltitudeDelta = 2000
       self.worstCaseHeadingDelta = 40

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

       # Add a conversion for heading, too:
       self.observation_minMaxes[1] = [-self.worstCaseHeadingDelta,
                                       self.worstCaseHeadingDelta]

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
          state, sim, numAltitudeStaggerLevels=20, numHeadingStaggerLevels=20,
          altitudeWorth=0.5, headingWorth=0.5)

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
