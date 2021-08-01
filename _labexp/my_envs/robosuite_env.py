import gym
from gym.spaces import Dict, Box
from gym.utils import seeding,EzPickle

import robosuite
from robosuite.wrappers import Wrapper, DomainRandomizationWrapper,GymWrapper
from robosuite.wrappers.domain_randomization_wrapper import DEFAULT_COLOR_ARGS, DEFAULT_CAMERA_ARGS, \
    DEFAULT_LIGHTING_ARGS, DEFAULT_DYNAMICS_ARGS
import numpy as np
from copy import deepcopy
from robosuite.controllers import load_controller_config, ALL_CONTROLLERS

#region from radsac
ROBOSUITE_ENVIRONMENTS = list(robosuite.ALL_ENVIRONMENTS)
ROBOSUITE_ROBOTS = list(robosuite.ALL_ROBOTS)
ROBOSUITE_CONTROLLERS = list(ALL_CONTROLLERS)

ROBOSUITE_DEFAULT_BASE_CONFIG = {
    'horizon': 1000,         # Every episode lasts for exactly horizon timesteps
    'ignore_done': True,     # True if never terminating the environment (ignore horizon)
    'reward_shaping': True,  # if True, use dense rewards.

    # How many control signals to receive in every simulated second. This sets the amount of simulation time
    # that passes between every action input (this is NOT the same as frame_skip)
    'control_freq': 10,

    # Optional observations (robot state is always returned)
    # if True, every observation includes a rendered image
    'use_camera_obs': False,
    # if True, include object (cube/etc.) information in the observation
    'use_object_obs': False,

    # Camera parameters
    'has_renderer': False,            # Set to true to use Mujoco native viewer for on-screen rendering
    'render_camera': 'frontview',     # name of camera to use for on-screen rendering
    'has_offscreen_renderer': False,
    'render_collision_mesh': False,   # True if rendering collision meshes in camera. False otherwise
    'render_visual_mesh': True,       # True if rendering visual meshes in camera. False otherwise
    'camera_names': 'agentview',      # name of camera for rendering camera observations
    'camera_heights': 84,             # height of camera frame.
    'camera_widths': 84,              # width of camera frame.
    'camera_depths': False,           # True if rendering RGB-D, and RGB otherwise.

    'hard_reset': False,              # Set to False as it doesn't play well with domain randomization

    # cube position for the observations under the 'sim2real_observations' folder
    # 'cube_pos_x_range': (-0.000001 + 1.17292297e-01, 0.000001 + 1.17292297e-01),
    # 'cube_pos_y_range': (-0.000001 + 3.33002579e-02, 0.000001 + 3.33002579e-02)
}

LIFT_DEFAULT_EXTRA_CONFIG = {
    # Collision
    'penalize_reward_on_collision': True,
    'end_episode_on_collision': False,
}

DR_WRAPPER_DEFAULT_CONFIG = {
    'randomize_color': True,
    'randomize_camera': True,
    'randomize_lighting': True,
    'randomize_dynamics': False,
    'color_randomization_args': DEFAULT_COLOR_ARGS,
    'camera_randomization_args': DEFAULT_CAMERA_ARGS,
    'lighting_randomization_args': DEFAULT_LIGHTING_ARGS,
    'dynamics_randomization_args': DEFAULT_DYNAMICS_ARGS,
    'randomize_on_reset': True,
    'randomize_every_n_steps_max': 20,
    'randomize_every_n_steps_min': 10,
}

DEFAULT_REWARD_SCALES = {
    'Lift': 2.25,
    'LiftLab': 2.25,
}


class RobosuiteEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, level, base_robosuite_config, robot, controller, extra_robosuite_config=None,
                 apply_dr=False, dr_wrapper_config=None, frame_skip=1,
                 obs_names_to_disable=('joint_vel', 'gripper_qvel')):
        # Validate arguments
        self.frame_skip = max(1, frame_skip)

        def validate_input(input, supported, name):
            if input not in supported:
                raise ValueError("Unknown Robosuite {0} passed: '{1}' ; Supported {0}s are: {2}".format(
                    name, input, ' | '.join(supported)
                ))
            return input

        self.env_id = validate_input(level, ROBOSUITE_ENVIRONMENTS, 'environment')
        self.robot = validate_input(robot, ROBOSUITE_ROBOTS, 'robot')
        if controller is not None:
            validate_input(controller, ROBOSUITE_CONTROLLERS, 'controller')
        self.controller = controller

        self.apply_dr = apply_dr
        self.obs_names_to_disable = obs_names_to_disable
        self._max_episode_steps = base_robosuite_config['horizon']

        self.config = deepcopy(base_robosuite_config)
        self.config['has_offscreen_renderer'] = self.config['has_offscreen_renderer'] or self.config['use_camera_obs']

        # Load and initialize environment
        extra_robosuite_config = extra_robosuite_config or {}
        self.config.update(extra_robosuite_config)

        if 'reward_scale' not in self.config and self.env_id in DEFAULT_REWARD_SCALES:
            self.config['reward_scale'] = DEFAULT_REWARD_SCALES[self.env_id]

        self.config['robots'] = self.robot
        controller_cfg = None
        if self.controller is not None:
            controller_cfg = robosuite.controllers.load_controller_config(default_controller=self.controller)
        self.config['controller_configs'] = controller_cfg

        self.env = robosuite.make(self.env_id, **self.config)

        # Disable requested observations.
        # Requirements on the requested names are "relaxed" - the requested name doesn't have to fully match
        # an available observation, only to be contained in one (or more).
        # So if, for example, the env contains 2 robots and so we have "robot0_joint_vel" and "robot1_joint_vel",
        # it's enough to specify "joint_vel" to disable both.
        available_observations = self.env.observation_names
        actual_obs_to_disable = set()
        for requested_obs in self.obs_names_to_disable:
            matches = [obs for obs in available_observations if requested_obs in obs]
            if not matches:
                raise ValueError("No observations containing the requested string '{}'. Available observations: {}"
                                 .format(requested_obs, available_observations))
            actual_obs_to_disable.update(matches)
        for obs in actual_obs_to_disable:
            self.env.modify_observable(obs, 'active', False)
            self.env.modify_observable(obs, 'enabled', False)

        self.env = Wrapper(self.env)

        if apply_dr:
            dr_wrapper_config = dr_wrapper_config or DR_WRAPPER_DEFAULT_CONFIG
            self.env = DomainRandomizationWrapper(self.env, **dr_wrapper_config)

        # Observation space
        dummy_obs = self._process_observation(self.env.observation_spec())
        obs_space_dict = {'measurements': Box(low=-np.inf, high=np.inf, shape=dummy_obs['measurements'].shape)}
        if self.config['use_camera_obs']:
            obs_space_dict['camera'] = Box(low=0, high=255, shape=dummy_obs['camera'].shape, dtype=np.uint8)
        self.observation_space = Dict(obs_space_dict)

        # Action space
        low, high = self.env.unwrapped.action_spec
        self.action_space = Box(low=low, high=high)


    def seed(self, seed=None):
        self.np_random_state, seed = seeding.np_random(seed)
        if self.apply_dr:
            self.env.random_state = self.np_random_state
        return [seed]

    def _process_observation(self, raw_obs):
        new_obs = {}

        # TODO: Support multiple cameras, this assumes a single camera
        camera_name = self.config['camera_names']
        camera_obs = raw_obs.get(camera_name + '_image', None)
        if camera_obs is not None:
            depth_obs = raw_obs.get(camera_name + '_depth', None)
            if depth_obs is not None:
                depth_obs = np.expand_dims(depth_obs, axis=2)
                camera_obs = np.concatenate([camera_obs, depth_obs], axis=2)
            new_obs['camera'] = camera_obs.transpose((2,0,1))

        measurements = raw_obs['robot0_proprio-state']
        object_obs = raw_obs.get('object-state', None)
        if object_obs is not None:
            measurements = np.concatenate([measurements, object_obs])
        new_obs['measurements'] = measurements

        return new_obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # We mimic the "action_repeat" mechanism of RobosuiteWrapper in Surreal.
        # Same concept as frame_skip, only returning the average reward across repeated actions instead
        # of the total reward.
        rewards = []
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break

        reward = np.mean(rewards)
        obs = self._process_observation(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        obs = self._process_observation(obs)
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            self.env.render()
        elif mode == 'rgb_array':
            img: np.ndarray = self.env.sim.render(camera_name=self.config['render_camera'],
                                                  height=512, width=512, depth=False)
            return np.flip(img, 0)
        else:
            super().render(mode=mode)


def create_environment(env_name, controller_name, pre_transform_image_size,
                       seed, apply_dr, obs_names_to_disable):
    import robosuite.utils.macros as macros
    # Must use this macro to get any randomization other then solid RGB color
    # for the robot arm
    # In turn, it means that the entire arm is randomized in the same way
    macros.USING_INSTANCE_RANDOMIZATION = True


    base_config = deepcopy(ROBOSUITE_DEFAULT_BASE_CONFIG)
    base_config['use_camera_obs'] = True
    base_config['has_offscreen_renderer'] = True
    base_config['camera_heights'] = pre_transform_image_size
    base_config['camera_widths'] = pre_transform_image_size
    base_config['horizon'] = 50
    base_config['ignore_done'] = False
    base_config['control_freq'] = 4
    base_config['render_camera'] = 'frontview'
    base_config['camera_names'] = 'labview'
    base_config['camera_depths'] = False

    extra_config = deepcopy(LIFT_DEFAULT_EXTRA_CONFIG)

    dr_color_args = deepcopy(DEFAULT_COLOR_ARGS)
    dr_color_args['geom_names'] = ['wall_front_visual', 'table_visual', 'robot0_link0_visual', 'gripper0_hand_visual',
                                   'mount0_plate_vis', 'cube_g0_vis']
    dr_color_args['randomize_local'] = False
    dr_color_args['randomize_skybox'] = False

    base_dr_wrapper_config = deepcopy(DR_WRAPPER_DEFAULT_CONFIG)
    base_dr_wrapper_config['color_randomization_args'] = dr_color_args
    base_dr_wrapper_config['randomize_dynamics'] = False
    base_dr_wrapper_config['randomize_every_n_steps_min'] = 1
    base_dr_wrapper_config['randomize_every_n_steps_max'] = 1  # once every step

    env = RobosuiteEnv(env_name, base_config, 'PandaLab', controller_name, extra_config, apply_dr=apply_dr,
                       dr_wrapper_config=base_dr_wrapper_config, obs_names_to_disable=obs_names_to_disable)

    if seed is not None:
        env.seed(seed)

    return env
#endregion

# the following class is a wrappper for using robosuite in garage
# todo: do we need a wrapper for gym.Env ? for example : pickling
#  check how other envs are wrapped in garage

class RobosuiteEnvGrg(gym.Env, EzPickle):
    metadata = {'render.modes': ['human', 'rgb_array']}
    # def __init__(self,env_name,robots,horizon,control_freq,reward_scale,
    #              hard_reset,ignore_done, ):
    def __init__(self, **env_config):
        EzPickle.__init__(self,**env_config)

        self.frame_skip = max(1, env_config.get('frame_skip',1))
        self.obs_names_to_disable = []
        self.apply_dr = False
        dr_wrapper_config=None

        self.config = deepcopy(env_config)

        self.env_id = self.config['env_name']

        self.horizon = self.config['horizon']

        self.config['has_offscreen_renderer'] = self.config['has_offscreen_renderer'] or self.config['use_camera_obs']

        if 'reward_scale' not in self.config and self.env_id in DEFAULT_REWARD_SCALES:
            self.config['reward_scale'] = DEFAULT_REWARD_SCALES[self.env_id]

        controller = self.config.pop('controller')
        if controller in set(ALL_CONTROLLERS):
            # A default controller
            controller_config = load_controller_config(default_controller=controller)
        else:   # a string to a custom controller
            controller_config = load_controller_config(custom_fpath=controller)
        self.config['controller_configs'] = controller_config

        self.env = robosuite.make(**self.config)

        # Disable requested observations.
        # Requirements on the requested names are "relaxed" - the requested name doesn't have to fully match
        # an available observation, only to be contained in one (or more).
        # So if, for example, the env contains 2 robots and so we have "robot0_joint_vel" and "robot1_joint_vel",
        # it's enough to specify "joint_vel" to disable both.
        available_observations = self.env.observation_names
        actual_obs_to_disable = set()
        for requested_obs in self.obs_names_to_disable:
            matches = [obs for obs in available_observations if requested_obs in obs]
            if not matches:
                raise ValueError("No observations containing the requested string '{}'. Available observations: {}"
                                 .format(requested_obs, available_observations))
            actual_obs_to_disable.update(matches)
        for obs in actual_obs_to_disable:
            self.env.modify_observable(obs, 'active', False)
            self.env.modify_observable(obs, 'enabled', False)

        # self.env = Wrapper(self.env)
        self.env = GymWrapper(self.env)

        # if self.apply_dr:
        #     dr_wrapper_config = dr_wrapper_config or DR_WRAPPER_DEFAULT_CONFIG
        #     self.env = DomainRandomizationWrapper(self.env, **dr_wrapper_config)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space


        # Observation space
        # dummy_obs = self._process_observation(self.env.observation_spec())
        # obs_space_dict = {'measurements': Box(low=-np.inf, high=np.inf, shape=dummy_obs['measurements'].shape)}
        # if self.config['use_camera_obs']:
        #     obs_space_dict['camera'] = Box(low=0, high=255, shape=dummy_obs['camera'].shape, dtype=np.uint8)
        # # currently ignoring the camera measurement. assuming always False
        # # self.observation_space = Dict(obs_space_dict)
        # self.observation_space = obs_space_dict['measurements']
        #
        # # Action space
        # low, high = self.env.unwrapped.action_spec
        # self.action_space = Box(low=low, high=high)


    def seed(self, seed=None):
        self.np_random_state, seed = seeding.np_random(seed)
        if self.apply_dr:
            self.env.random_state = self.np_random_state
        return [seed]

    def _process_observation(self, raw_obs):
        new_obs = {}

        # TODO: Support multiple cameras, this assumes a single camera
        camera_name = self.config.get('camera_names',None) or ''
        camera_obs = raw_obs.get(camera_name + '_image', None)
        if camera_obs is not None:
            depth_obs = raw_obs.get(camera_name + '_depth', None)
            if depth_obs is not None:
                depth_obs = np.expand_dims(depth_obs, axis=2)
                camera_obs = np.concatenate([camera_obs, depth_obs], axis=2)
            new_obs['camera'] = camera_obs.transpose((2,0,1))

        measurements = raw_obs['robot0_proprio-state']
        object_obs = raw_obs.get('object-state', None)
        if object_obs is not None:
            measurements = np.concatenate([measurements, object_obs])
        new_obs['measurements'] = measurements

        return new_obs['measurements']

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # We mimic the "action_repeat" mechanism of RobosuiteWrapper in Surreal.
        # Same concept as frame_skip, only returning the average reward across repeated actions instead
        # of the total reward.
        rewards = []
        for _ in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            rewards.append(reward)
            if done:
                break

        reward = np.mean(rewards)
        # obs = self._process_observation(obs)
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        # obs = self._process_observation(obs)
        return obs

    def render(self, mode='human'):
        if mode == 'human':
            self.env.render()
        elif mode == 'rgb_array':
            img: np.ndarray = self.env.sim.render(camera_name=self.config['render_camera'],
                                                  height=512, width=512, depth=False)
            return np.flip(img, 0)
        else:
            super().render(mode=mode)


class GrgGymWrapper(GymWrapper):
    def __init__(self,env):
        super().__init__(env=env)
        self.metadata={'render.modes': ['human', 'rgb_array']}
