from gym_jsbsim.task import Task
from gym_jsbsim.catalogs.catalog import Catalog as c
from gym import spaces
import math
import random
import numpy as np

"""
    @author Joe Williams

    A task in which the agent must perform steady, level flight maintaining its
    initial heading. Once the agent has been on heading/altitude for 30 seconds,
    a new goal heading and altitude is selected.

    This task was made before significant changes to task.py were made. It
    needs updating to function properly.
"""

class GetToChangingAltitudeAndHeadingTask(Task):

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

    def get_init_conditions(self):
       GetToChangingAltitudeAndHeadingTask.init_conditions = {
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
         c.target_heading_deg: self.currHeadingGoal,
         c.target_altitude_ft: self.currAltitudeGoal,
         c.fcs_throttle_cmd_norm: 0.8,
         c.fcs_mixture_cmd_norm: 1,
         c.gear_gear_pos_norm : 0,
         c.gear_gear_cmd_norm: 0,
         c.steady_flight:150}

       return GetToChangingAltitudeAndHeadingTask.init_conditions

    def getNewHeadingAndAltitudeTargets(self, currHeadingGoal, currAltitudeGoal, onCount):
        # Make sure the agent can't learn to always start turning a specific
        # direction or gaining/losing altitude, and that it can't correlate
        # heading change with altitude change.
        headingMult, altitudeMult = 1, 1
        if random.randint(0,1) == 0:
            headingMult = -1
        if random.randint(0,1) == 0:
            altitudeMult = -1

        # As onCount increases, we increase the amount of heading change and
        # altitude change required:
        newHeadingGoal = currHeadingGoal + \
          (headingMult * random.uniform(10.0 + (onCount * 5), 20.0 + (onCount * 10)))

        # Yes, headings will eventually swing around and we may end up asking
        # the agent to make a very slight turn.  That is okay; it should learn
        # to do that, too.
        newHeadingGoal = (round(newHeadingGoal) + 360) % 360

        newAltitudeGoal = currAltitudeGoal + \
          (altitudeMult * random.uniform(300.0 + (onCount * 200), 500.0 + (onCount * 400)))

        # If we're outside the acceptable range, flip the multiplier and try again:
        if newAltitudeGoal < self.minTargetAltitude or newAltitudeGoal > self.maxTargetAltitude:
            altitudeMult *= -1
            newAltitudeGoal = currAltitudeGoal + \
              (altitudeMult * random.uniform(300.0 + (onCount * 200), 500.0 + (onCount * 400)))

        # Still outside the acceptable range? Choose something guaranteed to be safe:
        if newAltitudeGoal < self.minTargetAltitude or newAltitudeGoal > self.maxTargetAltitude:
            newAltitudeGoal = random.uniform(self.minTargetAltitude, self.maxTargetAltitude)

        return newHeadingGoal, newAltitudeGoal

        ########################
        # This is returning bad heading / altitude goals (NaN).
        # Need to figure out why.
        ########################

    def __init__(self, floatingAction=True):
       super().__init__()
       # Variables we want to track and output at render time:
       self.mostRecentRewards = {}
       self.stopReason = None
       self.otherInfo = None
       self.simStopInfo = None

       # Selecting random altitudes to aim for, we need reasonable bounds:
       self.minTargetAltitude = 1000
       self.maxTargetAltitude = 20000

       # Count how many times we got on heading and altitude to reward the agent:
       self.onHeadingAndAltCount = 0
       self.onHeadingAndAltStartTime = None
       self.onAltitudeAndHeadingChangeTime = 30.0
       self.maxSimTime = 8000.0
       self.currHeadingGoal, self.currAltitudeGoal = self.getNewHeadingAndAltitudeTargets(100, 10000, 0)

       # Debug to count:
       self.stepCount = 0
       self.simTime = 0

       # How screwed is too screwed?
       self.worstCaseAltitudeDelta = 20000  # Wide range for possible large changes in goal altitude.
       self.worstCaseHeadingDelta = 180.0  # Always within this bound.

       # Hard version (Airforce standard):
       # self.onHeadingDifference = 5.0
       # self.onAltitudeDifference = 200.0

       # Medium:
       # self.onHeadingDifference = 7.5
       # self.onAltitudeDifference = 300.0

       # Easy:
       self.onHeadingDifference = 10.0
       self.onAltitudeDifference = 400.0

       self.floatingAction = floatingAction

       # Fill the min/max for our output conversion:
       self.observation_minMaxes = []
       for prop in GetToChangingAltitudeAndHeadingTask.state_var:
          self.observation_minMaxes.append([prop.min, prop.max])

       print(f"Prop min/maxes:\n{self.observation_minMaxes}")
       # The deltaAltitude is 40k.  Since we'll limit our aircraft differently,
       # change that variable:
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

       # Count how many times we got on heading and altitude to reward the agent:
       self.onHeadingAndAltCount = 0
       self.onHeadingAndAltStartTime = None
       self.onAltitudeAndHeadingChangeTime = 30.0
       self.maxSimTime = 8000.0
       self.currHeadingGoal, self.currAltitudeGoal = self.getNewHeadingAndAltitudeTargets(100, 10000, 0)

       # Debug to count:
       self.stepCount = 0
       self.simTime = 0

    def get_reward(self, state, sim):
        """Reward a plane for staying on altitude and heading."""

        d_alt = abs(sim.get_property_value(c.delta_altitude))
        altitudeReward = (self.worstCaseAltitudeDelta - d_alt) / self.worstCaseAltitudeDelta

        d_heading = abs(sim.get_property_value(c.delta_heading))
        headingReward = (self.worstCaseHeadingDelta - d_heading) / self.worstCaseHeadingDelta

        # Reward for your altitude and heading plus additional reward for how
        # many times you found the goal. This reward is larger than the best
        # possible on-alt and on-heading reward to encourage agents to get to
        # the target altitude and heading as frequently/quickly as possible.
        reward = (0.1 * altitudeReward) + (0.1 * headingReward) + (0.2 * float(self.onHeadingAndAltCount))

        self.mostRecentRewards = {
         'delta_alt': d_alt,
         'delta_heading': d_heading,
         'onCount': self.onHeadingAndAltCount,
         'reward': reward,
        }

        self.stepCount += 1
        self.simTime = sim.get_property_value(c.simulation_sim_time_sec)

        # Are we about to return NaN?
        if math.isnan(reward):
            print(f"""d_alt: {d_alt},
                    altitudeReward: {altitudeReward},
                    worstCaseAlt: {self.worstCaseAltitudeDelta},
                    d_heading: {d_heading},
                    headingReward: {headingReward},
                    worstCaseHeading: {self.worstCaseHeadingDelta},
                    onCount: {self.onHeadingAndAltCount},
                    reward: {reward}""")
            raise RuntimeError()

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
        if self.getOnHeadingAndAltitude(state, sim):
           simTime = sim.get_property_value(c.simulation_sim_time_sec)

           # First time on alt/heading?  Set the start time:
           if self.onHeadingAndAltStartTime is None:
               self.onHeadingAndAltStartTime = simTime
           # Have we been on alt/heading long enough to warrant a change?
           elif simTime - self.onHeadingAndAltStartTime >= self.onAltitudeAndHeadingChangeTime:
               self.onHeadingAndAltCount += 1
               # For now, choose random new heading and altitude goals. Later,
               # we may want to make sure that every time it gets a bit harder
               # by making the heading/altitude difference bigger.
               self.currHeadingGoal, self.currAltitudeGoal = \
                 self.getNewHeadingAndAltitudeTargets(self.currHeadingGoal,
                                                      self.currAltitudeGoal,
                                                      self.onHeadingAndAltCount)

               sim.set_property_value(c.target_altitude_ft, self.currAltitudeGoal)
               sim.set_property_value(c.target_heading_deg, self.currHeadingGoal)
               self.onHeadingAndAltStartTime = None

        else: # Not on alt/heading? Reset the counter.
           self.onHeadingAndAltStartTime = None

        # Run for a maximum of time or until we're way out of bounds.
        # Worst-case heading delta when selecting random new headings to aim for
        # is now 180, so that check will never fail. It is left in as a reminder
        # as we build other events to check for it.
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
