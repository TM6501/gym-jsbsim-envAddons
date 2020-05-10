from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np
import sys

"""
@author: Joseph Williams

This task isn't meant to be trained on, but rather as a test of already-trained
agents.  It will run the agent through a series of heading and altitude goal
changes for a very long time to ensure that the agent can handle every
variation of those changes at all altitudes.

Possible enhancements:
  - Determine why the convertObservation function very rarely finds NaN or Inf
    values.  Experimentally it seems to happen less than once every fifty runs,
    which makes it very hard to track down.
"""

class LongTestAltitudeAndHeadingChangesTask(Task):
    maxAltitude = 17000  # With a full fuel load, agent struggles to reach higher altitudes.
    minAltitude = 4000  # Can't go lower or sometimes the agent crashes on the way to its goal.

    def getNewHeadingAndAltitudeTargets(self):
       # Random choose altitude gain/loss and left/right turn sizes to
       # make sure the agent doesn't know what is coming.

       # This function will be called before __init__ finishes, so make sure
       # our starting values are set.
       try:
           self.currentAltitudeGoal
       except:
           self.currentAltitudeGoal = 10000

       try:
           self.currentHeadingGoal
       except:
           self.currentHeadingGoal = 100

       # 4 altitude choices: Big gain, small gain, small loss, big loss, no change.
       altChoice = random.randint(0, 4)
       gain = 0
       if altChoice == 0:  # Big gain:
           gain = random.uniform(5000.0, 8000.0)
       elif altChoice == 1:  # Small gain
           gain = random.uniform(1000.0, 3000.0)
       elif altChoice == 2:  # small loss
           gain = random.uniform(-1000.0, -3000.0)
       elif altChoice == 3:  # big loss
           gain = random.uniform(-5000.0, -8000.0)
       else:
           gain = 0

       # If we're already at min or max, we only go the other direction:
       if self.maxAltitude - self.currentAltitudeGoal <= 500.0:
           gain = -abs(gain)
       elif self.currentAltitudeGoal - self.minAltitude <= 500.0:
           gain = abs(gain)

       # Change the goal, but stay in bounds:
       self.currentAltitudeGoal += gain
       self.currentAltitudeGoal = min(self.maxAltitude, max(self.currentAltitudeGoal, self.minAltitude))

       if not (self.minAltitude <= self.currentAltitudeGoal <= self.maxAltitude):
           self.currentAltitudeGoal += (-2.0 * gain)

       # Heading change, same idea as the altitude change:  Big/small, left/right:
       turn = 0
       headingChoice = random.randint(0, 4)
       if headingChoice == 0:  # Big left
          turn = random.uniform(-50.0, -100.0)
       elif headingChoice == 1: # small Left
          turn = random.uniform(-20.0, -40.0)
       elif headingChoice == 2:  # small right.
          turn = random.uniform(20.0, 40.0)
       elif headingChoice == 3:  # big right
          turn = random.uniform(50.0, 100.0)
       else:  # No turn
          turn = 0

       self.currentHeadingGoal += turn
       # If we're now above/below 360.0, correct:
       if self.currentHeadingGoal > 360.0:
           self.currentHeadingGoal -= 360.0
       elif self.currentHeadingGoal < 0.0:
           self.currentHeadingGoal += 360.0

       return self.currentHeadingGoal, self.currentAltitudeGoal

    def get_initial_conditions(self):
       # Chang the start heading and altitude:
       headingGoal, altitudeGoal = self.getNewHeadingAndAltitudeTargets()

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
         # Full fuel, forever:
         c.propulsion_tank0_contents_lbs: 24000.0,  # 24k seems to be the max?
         c.propulsion_tank1_contents_lbs: 24000.0,
         c.propulsion_refuel: 1}

       return self.init_conditions

    def __init__(self, floatingAction=True):
       super().__init__()

       # We need to set tiny ranges to force precision from our agent:
       self.worstCaseAltitudeDelta = 2000
       # self.worstCaseHeadingDelta = 180.0

       # We've fallen into the bad habit of using worstCaseAltitudeDelta for
       # other things.  Need another variable:
       self.scenarioOverAltitudeDelta = 15000

       # Acceptable ranges to call the plane "on heading and altitude":
       self.onAltitudeDifference = 200.0
       self.onHeadingDifference = 5.0

       # Set a huge sim time to allow for many altitude and heading changes:
       self.maxSimTime = 30000.0  # 30k seconds

       # When did we first get on heading/altitude:
       self.onHeadingAndAltStartTime = None

       # How long do we need to stay on alt/heading to move on?
       self.onAltitudeAndHeadingChangeTime = 30.0
       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in self.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       # The deltaAltitude is 40k.  Since we'll limit our aircraft to different
       # delta-altitude, change that variable:
       self.observation_minMaxes[0] = [-self.worstCaseAltitudeDelta,
                                       self.worstCaseAltitudeDelta]

       # Add a conversion for heading, too:
       # As was noted in small-change-task, this doesn't seem necessary.
       # More experimentation is called for to determine why the altitude
       # min-max change is needed and this isn't.  [-40000, 40000] is a pretty
       # wide range...
       # self.observation_minMaxes[1] = [-self.worstCaseHeadingDelta,
       #                                 self.worstCaseHeadingDelta]

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

       # Although extremely rare, this does happen. Need to find out why:
       for i in range(len(retObs)):
           if math.isnan(retObs[i]) or retObs[i] == float('inf') or retObs[i] == float('-inf'):
               raise ValueError(f"Hit a bad value in observation: {observation}")

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

        # This function is mostly irrelevant because this environment is only
        # used for testing already-trained agents.  If this environment is ever
        # used for training, we'll need to modify the reward to account for
        # how many times the agent got on altitude/heading to give it proper
        # rewards for that behavior.
        return reward

    def getOnHeadingAndAltitude(self, state, sim):
        d_alt = math.fabs(sim.get_property_value(c.delta_altitude))
        d_heading = math.fabs(sim.get_property_value(c.delta_heading))

        if d_alt <= self.onAltitudeDifference and d_heading <= self.onHeadingDifference:
            return True
        else:
            return False

    def is_terminal(self, state, sim):
        # If we've been on-altitude and on-heading for a while, change the
        # targets:
        simTime = sim.get_property_value(c.simulation_sim_time_sec)
        if self.getOnHeadingAndAltitude(state, sim):
            # First time on alt/heading?  Set the start time:
            if self.onHeadingAndAltStartTime is None:
                 self.onHeadingAndAltStartTime = simTime
            # Have we been on alt/heading long enough to warrant a change?
            elif simTime - self.onHeadingAndAltStartTime >= self.onAltitudeAndHeadingChangeTime:
                # Choose new altitude and heading goals:
                 self.currHeadingGoal, self.currAltitudeGoal = \
                   self.getNewHeadingAndAltitudeTargets()

                 # Send to error out so that it doesn't get recorded in the CSV file:
                 print(f"Resetting goals: {self.currentAltitudeGoal}, {self.currentHeadingGoal}", file=sys.stderr)

                 sim.set_property_value(c.target_altitude_ft, self.currAltitudeGoal)
                 sim.set_property_value(c.target_heading_deg, self.currHeadingGoal)
                 self.onHeadingAndAltStartTime = None

        else: # Not on alt/heading? Reset the counter.
            if self.onHeadingAndAltStartTime is not None:
                print(f"Resetting counter. Was at: {simTime - self.onHeadingAndAltStartTime}.", file=sys.stderr)
            self.onHeadingAndAltStartTime = None

        # Run for a maximum of time or until we're way out of bounds:
        outOfTime = sim.get_property_value(c.simulation_sim_time_sec) >= self.maxSimTime
        dAlt = math.fabs(sim.get_property_value(c.delta_altitude)) >= self.scenarioOverAltitudeDelta
        extremeState = bool(sim.get_property_value(c.detect_extreme_state))
        tooLow = sim.get_property_value(c.position_h_sl_ft) < 100.0

        retVal =  outOfTime or dAlt or extremeState or tooLow
        if retVal and not outOfTime:
            print(f"Terminal. dAlt: {dAlt}, extremeState: {extremeState}, tooLow: {tooLow}", file=sys.stderr)

        return retVal
