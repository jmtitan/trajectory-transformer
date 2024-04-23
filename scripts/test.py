import json
import pdb
from os.path import join
import numpy as np
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    our_beam_plan,
    make_prefix,
    extract_actions,
    update_context,
    MCTS
)

class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.offline'

#######################
######## setup ########
#######################

args = Parser().parse_args('plan')

#######################
####### models ########
#######################

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)

#######################
####### dataset #######
#######################


env = datasets.load_environment(args.dataset)
# renderer = utils.make_renderer(args)

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)

#######################
###### main loop ######
#######################

observation = env.reset()
total_reward = 0

## observations for rendering
rollout = [observation.copy()]

## previous (tokenized) transitions for conditioning transformer
context = []


## basic settings
T = args.beam_width
expands = args.beam_width * 2
observation = preprocess_fn(observation)
prefix = make_prefix(discretizer, context, observation, args.prefix_context)
time_list = []

## experiment on Comparison times

# for t in range(1, T+1):
#     timer = utils.timer.Timer()
#     ## sample sequence from model beginning with `prefix`
#     sequence = our_beam_plan(
#         gpt, value_fn, prefix,
#         args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim, t,
#         discount, args.max_context_transitions, verbose=args.verbose,
#         k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act
#     )

#     rec_t = timer()
#     time_list.append(rec_t)
#     print(
#         f'[ Comparison times ] t: {t} / {T} | '
#         f'time: {rec_t:.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
#     )

# np.savetxt("expand_times.csv", np.array(time_list), delimiter=",")

## experiment on expand times
# for expand in range(1, expands):
#     timer = utils.timer.Timer()
#     ## sample sequence from model beginning with `prefix`
#     sequence = our_beam_plan(
#         gpt, value_fn, prefix,
#         args.horizon, args.beam_width, expand, observation_dim, action_dim, 2,
#         discount, args.max_context_transitions, verbose=args.verbose,
#         k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act
#     )

#     rec_t = timer()
#     time_list.append(rec_t)
#     print(
#         f'[ expand_times ] t: {expand} / {T} | '
#         f'time: {rec_t:.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
#     )

# np.savetxt("expand_times.csv", np.array(time_list), delimiter=",")


# experiment on avg time
for t in range(1, 20):
    timer = utils.timer.Timer()
    ## sample sequence from model beginning with `prefix`
    sequence = our_beam_plan(
        gpt, value_fn, prefix,
        args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim, 2,
        discount, args.max_context_transitions, verbose=args.verbose,
        k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act
    )

    rec_t = timer()
    time_list.append(rec_t)
    print(
        f'[ epoch ] t: {t} / {T} | '
        f'time: {rec_t:.2f} | {args.dataset} | {args.exp_name} | {args.suffix}\n'
    )
print( f'[ avg time ] time: {sum(time_list)/20:.2f} ')