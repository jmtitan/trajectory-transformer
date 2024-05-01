import json
import time
import numpy as np
from os.path import join
import pandas as pd
import trajectory.utils as utils
import trajectory.datasets as datasets
from trajectory.search import (
    beam_plan,
    A_star_beam_plan,
    Q_beam_plan,
    Best_first_beam_plan,
    make_prefix,
    extract_actions,
    update_context,
    MCTS,
    q_function_guided_action
)
from trajectory.heuristic.iql import load_iql


beam_function = [
    'BEAM_PLAN',
    'week_MCTS',
    'Q_BEAM_PLAN',
    'Best_FIRST_BEAM_PLAN',
    'A_STAR_BEAM_PLAN'
]


class Parser(utils.Parser):
    dataset: str = 'maze2d-umaze-v1'
    config: str = 'config.offline'

args = Parser().parse_args('plan')

dataset = utils.load_from_config(args.logbase, args.dataset, args.gpt_loadpath,
        'data_config.pkl')

gpt, gpt_epoch = utils.load_model(args.logbase, args.dataset, args.gpt_loadpath,
        epoch=args.gpt_epoch, device=args.device)


env = datasets.load_environment(args.dataset)
# renderer = utils.make_renderer(args)
timer = utils.timer.Timer()

discretizer = dataset.discretizer
discount = dataset.discount
observation_dim = dataset.observation_dim
action_dim = dataset.action_dim

value_fn = lambda x: discretizer.value_fn(x, args.percentile)
preprocess_fn = datasets.get_preprocess_fn(env.name)


# load heuristic q function
iql_path = './logs/maze2d-umaze-v1/iql-10e6/final.pt'
iql = load_iql(observation_dim, action_dim, iql_path)

### planing
for test_beam in beam_function:
    observation = env.reset()

    ## manual set starting point
    qpos = np.array([1.9036427614959943,3.056446507881826])
    qvel = np.array([-0.1034511263986435,0.014453362443003024])
    env.set_state(qpos, qvel)
    observation = env.state_vector()

    env_goals = env.empty_and_goal_locations
    total_reward = 0
    cur_goal_idx = 6

    ## observations for rendering
    rollout = []
    goals = []
    tol = 0.07

    ## previous (tokenized) transitions for conditioning transformer
    context = []

    st = time.time()
    T = env.max_episode_steps
    for t in range(T):
        goal = env._target
        rollout.append(observation.tolist())
        rollout[-1].extend(goal)


        observation = preprocess_fn(observation) # one starting point (x,y, vx, vy) -> tokenizer(x,y,vx,vy)

        if t % args.plan_freq == 0:
            ## concatenate previous transitions and current observations to input to model
            prefix = make_prefix(discretizer, context, observation, args.prefix_context)

            # sample sequence from model beginning with `prefix`
            
            if test_beam == "Q_BEAM_PLAN":
                sequence = Q_beam_plan(
                    gpt, iql, discretizer.reconstruct_torch, prefix,
                    args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                    discount, args.max_context_transitions, verbose=args.verbose,
                    k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                )
            elif test_beam == "BEAM_PLAN":
                sequence = beam_plan(
                    gpt, value_fn, prefix,
                    args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                    discount, args.max_context_transitions, verbose=args.verbose,
                    k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                )
            elif test_beam == "week_MCTS":
                sequence = MCTS(
                    gpt, value_fn, prefix,
                    args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                    discount, args.max_context_transitions, verbose=args.verbose,
                    k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                )
            elif test_beam == "A_STAR_BEAM_PLAN":
                sequence = A_star_beam_plan(
                    gpt, iql, discretizer.reconstruct_torch, prefix,
                    args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                    discount, args.max_context_transitions, verbose=args.verbose,
                    k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                )
            elif test_beam == "Best_FIRST_BEAM_PLAN":
                sequence = Best_first_beam_plan(
                    gpt, value_fn, prefix,
                    args.horizon, args.beam_width, args.n_expand, observation_dim, action_dim,
                    discount, args.max_context_transitions, verbose=args.verbose,
                    k_obs=args.k_obs, k_act=args.k_act, cdf_obs=args.cdf_obs, cdf_act=args.cdf_act,
                )
        else:
            sequence = sequence[1:]  

        ## [ horizon x transition_dim ] convert sampled tokens to continuous trajectory
        sequence_recon = discretizer.reconstruct(sequence)  #sequence_recon = (x,y,vx,vy,ax,ay,...)

        ## [ action_dim ] index into sampled trajectory to grab first action
        action = extract_actions(sequence_recon, observation_dim, action_dim, t=0)
        # action = q_function_guided_action(sequence_recon, observation_dim, action_dim, iql)

        ## execute action in environment
        reward = 0
        if abs(observation[0] - goal[0]) < tol and abs(observation[1] - goal[1]) < tol:
            reward = 1.0
            cur_goal_idx += 1
            env.set_target(env_goals[cur_goal_idx%7])

        next_observation, _, terminal, _ = env.step(action)

        ## update return
        total_reward += reward
        score = env.get_normalized_score(total_reward)

        rollout[-1].extend([reward, score])


        ## update rollout observations and context transitions
        context = update_context(context, discretizer, observation, action, reward, args.max_context_transitions)

        print(
            f'[ plan ] t: {t} / {T} | r: {reward:.2f} | R: {total_reward:.2f} | score: {score:.4f} | '
            f'time: {timer():.2f} | {args.dataset} | {args.exp_name} | goal: {goal} | {args.suffix}\n'
        )


        observation = next_observation
    nd = time.time()
    print('='*100, f'\n{test_beam} avg time cost per step: {(nd - st)/T}\n', '='*100)
    ## save result as a json file
    json_path = join(args.savepath, test_beam + '-rollout.json')
    json_data = {'score': score, 'step': t, 'avg time': (nd - st)/T, 'return': total_reward, 'term': terminal, 'gpt_epoch': gpt_epoch}
    json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)


    df = pd.DataFrame(rollout, columns=['x','y','vx','vy','goalx','goaly', 'reward', 'score'])
    df.to_csv(join(args.savepath, test_beam + '-data.csv'))
