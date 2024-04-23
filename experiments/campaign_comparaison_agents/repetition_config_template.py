'''import numpy as np
import exputils as eu
from human_aware_navigation_rl.agent import *
from human_aware_navigation_rl.env.mpi_env.mpi_env import MpiEnv, eval_function, log_function, memory_storage_analysis_function
from human_aware_navigation_rl.networks import *
from human_aware_navigation_rl.networks.utils.grid_autoencoders.grid_autoencoder import GridAutoEncoder
from human_aware_navigation_rl.networks.utils.pre_nav_encoders.pre_nav_encoder import PreNavEncoder

config = eu.AttrDict(

    seed = 505 + <repetition_id>,

    n_max_episodes = 700,
    n_max_episodes_eval = 20,
    n_max_steps = np.inf,
    is_update_agent = True,

    log_functions=[],

    memory_storage_analysis = False,

    agent = eu.AttrDict(
        cls=<agent>,

        batch_size = <batch_size>,
        discount_factor = <discount_factor>,
        soft_target_update = <soft_target_update>,  # Target smoothing coefficient(τ)
        learning_rate = <learning_rate>,
        network_update = <network_update>, # Number of steps between each backprop
        nb_gradient_steps = <nb_gradient_steps>,  # Number of backprop per update step
        policy_noise = <policy_noise>,

        # Parameters for actor network
        # Only for TD3
        policy_delay = <policy_delay>,
    ),


    # Environment parameters
    env = eu.AttrDict(
        cls=MpiEnv,
        action_space_type = 'continuous',
        # Parameters about furnitures
        n_round_tables = 4,
        tables_positions = [[0, -3], [-3, 0], [3, 0], [0, 3]],
        table_radius = [0.5, 0.5, 0.5, 0.5],
        # Parameters about ARI
        random_start_position = 1,
        random_goal_position = 1,
        raycast_field_of_view = 360,
        raycast_angle_between_two_ray = 10,
        raycast_max_range = 3,
        probability_appearance_furnitures = 0.5,
        desired_grid_size = (60,60), 

        
        # Reward parameters
        reward_distance_progress_weight = 10,
        reward_safety_progress_weight = 0,
        reward_curving_weight = 0,
        reward_per_step = -5,
        reward_goal_reached = 500,
        reward_collision = -500,
        distance_progress_potential=False,

        
    ),

    env_eval_0 = eu.AttrDict(
        # Fixed table and moving start/goal positions
        # Only insert parameter that has to be modified during the evaluation
        probability_appearance_furnitures = 1, # probability to have furniture in the episode

    )
)
'''