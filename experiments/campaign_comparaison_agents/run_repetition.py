'''#!/usr/bin/env python
from human_aware_navigation_rl.exp.evaluation import Evaluation
from human_aware_navigation_rl.exp.training import run_training
import exputils as eu
from repetition_config import config
from human_aware_navigation_rl.env.mpi_env.mpi_env import MpiEnv, eval_function, log_function, memory_storage_analysis_function
import torch
from copy import deepcopy

torch.set_num_threads(1)

log_state = eu.AttrDict()
config.log_functions = [log_function]
# create the eval env
log_state.evaluation_env0 =  Evaluation(config=config, eval_id = 0, config_eval = config.env_eval_0)

log, agent, env = run_training(
    config=config,
    log_state=log_state,
    log_functions = [eval_function, log_function],
)

log.save()
env.simulation.close()
agent.save_checkpoint(ckpt_path = "checkpoints/final_")'''