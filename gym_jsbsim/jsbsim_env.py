import gym
import numpy as np
from gym_jsbsim.simulation import Simulation


class JSBSimEnv(gym.Env):

    """
    A class wrapping the JSBSim flight dynamics module (FDM) for simulating
    aircraft as an RL environment conforming to the OpenAI Gym Env
    interface.

    An JsbSimEnv is instantiated with a Task that implements a specific
    aircraft control task with its own specific observation/action space and
    variables and agent_reward calculation.

    ATTRIBUTION: this class implements the OpenAI Gym Env API. Method
    docstrings have been adapted or copied from the OpenAI Gym source code.
    """

    metadata = {'render.modes': ['human', 'csv']}

    def __init__(self, task):
        """

        Constructor. Init some internal state, but JSBSimEnv.reset() must be

        called first before interacting with environment.

        :param task: the Task for the task agent is to perform

        """

        self.sim = None
        self.task = task()

        self.observation_space = self.task.get_observation_space()  # None
        self.action_space = self.task.get_action_space()  # None

        self.state = None

    def step(self, action=None):
        """

        Run one timestep of the environment's dynamics. When end of

        episode is reached, you are responsible for calling `reset()`

        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).



        :param action: np.array, the agent's action, with same length as action variables.

        :return:

            state: agent's observation of the current environment

            reward: amount of reward returned after previous action

            done: whether the episode has ended, in which case further step() calls are undefined

            info: auxiliary information

        """
        self.task.notifyAction(action)

        # If we aren't asking the agent to control the rudder, append 0.0 for it:
        if len(action) == 3:
            action = np.append(action, 0.0)

        self.state = self.make_step(action)

        reward, done, info = self.task.get_reward(self.state, self.sim), self.is_terminal(), {}
        state = self.state
        # if done:
        #    state = self._get_clipped_state() # returned state should be in observation_space

        # print(f"*******************\n\tAction: {action}\n\tState: {state}\n\tReward: {reward}\n********************")


        return state, reward, done, info

    def make_step(self, action=None):
        """

        Calculates new state.


        :param action: array of floats, the agent's last action

        :return: observation: array, agent's observation of the environment state


        """

        # take actions
        if action is not None:
            self.sim.set_property_values(self.task.get_action_var(), action)

        # run simulation
        self.sim.run()

        return self.get_observation()

    def reset(self):
        """

        Resets the state of the environment and returns an initial observation.

        :return: array, the initial observation of the space.

        """
        if self.sim:
            self.sim.close()

        self.sim = Simulation(aircraft_name=self.task.aircraft_name,
                              init_conditions=self.task.get_initial_conditions(),
                              jsbsim_freq=self.task.get_jsbsim_freq(),
                              agent_interaction_steps=self.task.get_agent_interaction_steps())

        self.state = self.get_observation()

        self.observation_space = self.task.get_observation_space()

        self.action_space = self.task.get_action_space()

        return self.state

    def is_terminal(self):
        """

        Checks if the state is terminal.

        :return: bool

        """
        is_not_contained = not self.observation_space.contains(self.state)

        return is_not_contained or self.task.is_terminal(self.state, self.sim)

    def render(self, mode='human', **kwargs):
        """Renders the environment.

        The set of supported modes varies per environment. (And some

        environments do not support rendering at all.) By convention,

        if mode is:

        - human: print on the terminal
        - csv: output to cvs files

        Note:

            Make sure that your class's metadata 'render.modes' key includes

              the list of supported modes. It's recommended to call super()

              in implementations to use the functionality of this method.



        :param mode: str, the mode to render with
        """
        # self.task.render(self.sim, mode=mode, **kwargs)
        self.task.render(self.sim, **kwargs)

        # This version of output doesn't work, let the task spit out something useful:
        # output = self.task.get_output()
        #
        # if mode == 'human':
        #     pass
        # elif mode == 'csv':
        #     pass
        # else:
        #     pass


    def seed(self, seed=None):
        """

        Sets the seed for this env's random number generator(s).

        Note:

            Some environments use multiple pseudorandom number generators.

            We want to capture all such seeds used in order to ensure that

            there aren't accidental correlations between multiple generators.

        Returns:

            list<bigint>: Returns the list of seeds used in this env's random

              number generators. The first value in the list should be the

              "main" seed, or the value which a reproducer should pass to

              'seed'. Often, the main seed equals the provided 'seed', but

              this won't be true if seed=None, for example.

        """
        return

    def close(self):
        """ Cleans up this environment's objects

        Environments automatically close() when garbage collected or when the

        program exits.

        """
        if self.sim:
            self.sim.close()

    def get_observation(self):
        """
        get state observation from sim.

        :return: NamedTuple, the first state observation of the episode

        """
        obs_list = self.sim.get_property_values(self.task.get_observation_var())
        obs = tuple([np.array([obs]) for obs in obs_list])
        return self.task.convertObservation(obs)

    def get_sim_time(self):
        """ Gets the simulation time from sim, a float. """
        return self.sim.get_sim_time()

    def get_state(self):
        raise NotImplementedError("To be implemented")
        #return self.sim.get_sim_state()

    def _get_clipped_state(self):
        clipped = [
            np.clip(self.state[i], o.low, o.high) if self.task.state_var[i].clipped else self.state[i]
            for i, o in enumerate(self.observation_space)
        ]
        return tuple(clipped)

    def set_state(self, state):
        raise NotImplementedError("To be implemented")
        #self.sim.set_sim_state(state)
        #self.state = self.get_observation()
