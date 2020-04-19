from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""Since agents seem to have trouble making big jumps in heading and altitude,
we'll ask them to make a series of small adjustments until they reach their
goal.  The task is for testing that capability; it may not serve well for
training."""

class GetToAltitudeAndHeadingSequentialTask(Task):

    headingChangeStepSize = 10.0
    altitudeChangeStepSize = 1000.0

    # The values required to be considered successful:
    onGoalAltitudeDiff = 200.0
    onGoalHeadingDiff = 5.0

    # Number of seconds to remain on goal before the goal changes:
    onGoalTime = 20.0

    # def getStartHeadingAndAltitudeChanges(self):
    #    # We have a large heading/altitude change goal:
    #
    #    # Choose our full altitude change:
    #    altitudeChoice = random.randint(-1, 1)
    #
    #    altitude = 0
    #    if altitudeChoice == -1:
    #        altitude = random.uniform(-4500.0, -3500.0)
    #    elif altitudeChoice == 0:
    #        altitude = random.uniform(-200.0, 200.0)
    #    else:
    #        altitude = random.uniform(3500.0, 4500.0)
    #
    #    # Choose our full heading change:
    #    headingChoice = random.randint(-1, 1)
    #
    #    heading = 0
    #    if headingChoice == -1:
    #        heading = random.uniform(-85.0, -95.0)
    #    elif headingChoice == 0:
    #        heading = random.uniform(-10.0, 10.0)
    #    else:
    #        heading = random.uniform(85.0, 95.0)
    #
    #    return heading, altitude

    def getStartHeadingAndAltitudeChanges(self):
       # We have a large heading/altitude change goal:

       # Choose our full altitude change:
       altitudeChoice = random.randint(0, 1)

       altitude = 0
       if altitudeChoice == 0:
           altitude = random.uniform(-4500.0, -3500.0)
       else:
           altitude = random.uniform(3500.0, 4500.0)

       # Choose our full heading change:
       headingChoice = random.randint(0, 1)

       heading = 0
       if headingChoice == 0:
           heading = random.uniform(-85.0, -95.0)
       else:
           heading = random.uniform(85.0, 95.0)

       return heading, altitude

    def getNextGoal(self, currentValue, finalGoal, stepSize):
        """Determine the next step between our current goal and the final
        goal."""
        # If we're within striking distance (or already there) return the
        # final goal:
        if abs(currentValue - finalGoal) <= stepSize:
            return finalGoal

        # Otherwise, take a step in the necessary direction:
        if finalGoal < currentValue:
            return currentValue - stepSize
        else:
            return currentValue + stepSize

    def getNextAltitudeAndHeadingGoals(self):
        """Get the next sub-goal.  We only take steps of up to 1000 feet of
        altitude and 10 degrees of heading."""
        newHeadingGoal = self.getNextGoal(self.currentHeadingGoal,
                                          self.finalHeadingGoal,
                                          self.headingChangeStepSize)

        newAltitudeGoal = self.getNextGoal(self.currentAltitudeGoal,
                                           self.finalAltitudeGoal,
                                           self.altitudeChangeStepSize)

        return newHeadingGoal, newAltitudeGoal

    def get_initial_conditions(self):
       # Set our initial goals:
       headingDiff, altitudeDiff = self.getStartHeadingAndAltitudeChanges()

       self.currentHeadingGoal = 100  # Our start heading.
       self.finalHeadingGoal = 100 + headingDiff

       self.currentAltitudeGoal = 10000 # Our start altitude.
       self.finalAltitudeGoal = 10000 + altitudeDiff

       initHeading, initAltitude = self.getNextAltitudeAndHeadingGoals()

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

       # We must use the same altitude and heading worst-case values as
       # GetToAltitudeAndHeadingSmallTask in order to allow the agent trained
       # there to function here:
       self.worstCaseAltitudeDelta = 2000
       self.worstCaseHeadingDelta = 20
       self.onHeadingAndAltStartTime = None

       # How far can we get off of our goal before we start getting negative rewards?
       # For now, we're not using these:
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

    def getOnHeadingAndAltitude(self, state, sim):
        d_alt = math.fabs(sim.get_property_value(c.delta_altitude))
        d_heading = math.fabs(sim.get_property_value(c.delta_heading))

        if d_alt <= self.onGoalAltitudeDiff and \
           d_heading <= self.onGoalHeadingDiff:
            return True
        else:
            return False

    # Define our own terminal function to continue changing the targets:
    def is_terminal(self, state, sim):
        timeOut = sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime
        extremeOut = bool(sim.get_property_value(c.detect_extreme_state))

        # Give the altitude and heading failure values a bit of extra leeway
        # due to the extremely strict training:
        altitudeOut = math.fabs(sim.get_property_value(c.delta_altitude)) >= self.worstCaseAltitudeDelta + 1500.0
        headingOut = math.fabs(sim.get_property_value(c.delta_heading)) >= self.worstCaseHeadingDelta + 15.0


        isTerminal = timeOut or altitudeOut or headingOut or extremeOut

        # If we aren't done, check to see if it is time to change goals:
        if not isTerminal:
            simTime = sim.get_property_value(c.simulation_sim_time_sec)
            onGoal = self.getOnHeadingAndAltitude(state, sim)

            if self.onHeadingAndAltStartTime is None:
                self.onHeadingAndAltStartTime = simTime
            # We've been on goal long enough, change it:
            elif simTime - self.onHeadingAndAltStartTime > self.onGoalTime:
                self.currentHeadingGoal, self.currentAltitudeGoal = self.getNextAltitudeAndHeadingGoals()
                # print(f"New goals: {self.currentHeadingGoal},  {self.currentAltitudeGoal}")
                # print(f"Final goals: {self.finalHeadingGoal}, {self.finalAltitudeGoal}")

                sim.set_property_value(c.target_altitude_ft, self.currentAltitudeGoal)
                sim.set_property_value(c.target_heading_deg, self.currentHeadingGoal)
                self.onHeadingAndAltStartTime = None

        return isTerminal

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
