import numpy as np
import torch
from .. import utils
from .sampling import sample_n, sort_2d, sample,sample_n_topp

REWARD_DIM = VALUE_DIM = 1


    
@torch.no_grad()
def A_star_beam_plan(
    model, iql,reconstruct_fun,  x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,comparison_time,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''


    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }


    ## construct discount tensors for estimating values
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)
    # rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)

    ## init big_Q
    big_Q = -torch.inf * torch.ones(n_steps+1) # beam_width * 1
    d = {}
    t = 0
    sh_0 = 0 # final score 
    r_0 = torch.zeros(n_steps + 1, device=x.device)
    init_heap = {sh_0 : (x, r_0)}
    big_Q[t] = sh_0
    d[t] = init_heap
    POPS = {}


    ## logging
    # progress = utils.Progress(n_steps) if verbose else utils.Silent()
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    while n_steps not in d.keys():
        queue_sort = torch.argsort(big_Q)
        t = int(queue_sort[-1])
        top_sh = next(iter(d[t]))
        (seq, rewards) = d[t][top_sh]            # pop out the best seq in t step: d[t]
        del d[t][top_sh]                           # del the best seq in d[t]


        if len(d[t]) == 0:
            big_Q[t] = -torch.inf
        else:
            best_remaining_sh = next(iter(d[t]))
            big_Q[t] = best_remaining_sh

 
        ## init POPS
        if t not in POPS:
            POPS[t] = 0
        
        ## check beam size beam_width = 128, n_steps = 15
        if POPS[t] >= beam_width or t > n_steps:
            continue
        POPS[t] += 1


        seq = seq.repeat(beam_width * n_expand, 1)
        rewards = rewards.repeat(beam_width * n_expand, 1)

        # actions
        seq , _ = sample_n(model, seq, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)
        # reward
        seq, r_prob = sample_n(model, seq, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)
        
        gap = observation_dim + action_dim + REWARD_DIM + VALUE_DIM


        raw_x =  reconstruct_fun(seq[:, -gap:])
        obs = raw_x[:, :observation_dim]
        act = raw_x[:, observation_dim: observation_dim + action_dim]
        q_vals = iql.qf(obs, act)
        
        rewards[:, t] = q_vals
        sh = (rewards * discounts).sum(dim=-1) 
        

        # observation
        if t < n_steps - 1:
            seq, _ = sample_n(model, seq, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        sh, ind = torch.sort(sh, descending=True)
        rewards = rewards[ind]
        seq = seq[ind]

        for i in range(beam_width):
            if t+1 in d.keys():
                if sh[i] not in list(d[t+1].keys()): # clear repeated path
                    d[t+1][sh[i]] = (seq[i], rewards[i])
            else: #we need toq initialize B_t+1
                init_heap = {sh[i]: (seq[i], rewards[i])}
                d[t+1] = init_heap

        best_sh_t1 = next(iter(d[t+1]))
        big_Q[t+1] = best_sh_t1
        # inital version
        # for sh_i, r_i, seq_i in zip(torch.chunk(sh, beam_width * n_expand, dim=0), 
        #                        torch.chunk(rewards, beam_width * n_expand, dim = 0),
        #                         torch.chunk(seq, beam_width * n_expand, dim = 0)):
        #     sh_i = sh_i[0].item()
        #     seq_i = seq_i[0]
        #     r_i = r_i[0]
        #     if t+1 in d.keys():
        #         if len(d[t+1]) < beam_width:
        #             d[t+1].append((sh_i, (seq_i, r_i)))
        #             d[t+1] = sorted(d[t+1], key=lambda x: x[0])
        #         else: #We have to check for dominated sequences
        #             if d[t+1][0][0] < sh_i: #We replace them
        #                 d[t+1][0] =  (sh_i, (seq_i, r_i))
        #                 d[t+1] = sorted(d[t+1], key=lambda x: x[0])
        #     else: #we need toq initialize B_t+1
        #         init_heap = [(sh_i, (seq_i, r_i))]
        #         d[t+1] = init_heap
                
                ## Q -> (k, )
        
        best_sh_t1 = next(iter(d[t+1]))
        big_Q[t+1] = best_sh_t1

        progress.update({
            'x': list(x.shape),
            'vmin': sh.min(), 'vmax': sh.max(),
            'qmin': q_vals.min(), 'qmax': q_vals.max(),
            'discount': discount
        })

    progress.stamp()

    best_seq = d[n_steps][next(iter(d[n_steps]))][0]
    best_seq = best_seq.view(-1, transition_dim)
    return best_seq


@torch.no_grad()
def MCTS(
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None, p=0.75
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    inp = x.clone()

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    

    ## construct discount tensors for estimating values
    rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)
   

    ## repeat input for search
    x = x.repeat(beam_width, 1)
    ## logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    

    for t in range(n_steps):

        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n_topp(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## logging
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            'discount': discount
        })

    progress.stamp()

    ## [ batch_size=256 x (n_context + n_steps) x transition_dim=8 ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## return best sequence (15 \times 8)
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence

@torch.no_grad()
def beam_plan(
    model, value_fn, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    inp = x.clone()

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    

    ## construct discount tensors for estimating values
    rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)
    discounts = discount ** torch.arange(n_steps + 1, device=x.device)
   

    ## repeat input for search
    x = x.repeat(beam_width, 1)
    ## logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    

    for t in range(n_steps):

        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)

        
        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_probs)

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## logging
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            'discount': discount
        })

    progress.stamp()

    ## [ batch_size=256 x (n_context + n_steps) x transition_dim=8 ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## return best sequence (15 \times 8)
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence




@torch.no_grad()
def Q_beam_plan(
    model, q_model, reconstruct_fun, x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''

    inp = x.clone()

    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }

    

    ## construct discount tensors for estimating values
    rewards = torch.zeros(beam_width, n_steps, device=x.device)
    discounts = discount ** torch.arange(n_steps, device=x.device)
   

    ## repeat input for search
    x = x.repeat(beam_width, 1)
    ## logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()

    

    for t in range(n_steps):

        ## repeat everything by `n_expand` before we sample actions
        x = x.repeat(n_expand, 1)
        rewards = rewards.repeat(n_expand, 1)

        ## sample actions
        x, _ = sample_n(model, x, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)


        ## sample reward and value estimate
        x, r_probs = sample_n(model, x, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        gap = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
        
        raw_x =  reconstruct_fun(x[:, -gap:])
    
        obs = raw_x[:, :observation_dim]
        act = raw_x[:, observation_dim: observation_dim + action_dim]
        q_vals = q_model.qf(obs, act)


        ## update rewards tensor
        rewards[:, t] = q_vals

        ## estimate values using rewards up to `t` and terminal value at `t`
        values = (rewards * discounts).sum(dim=-1)

        ## get `beam_width` best actions
        values, inds = torch.topk(values, beam_width)

        ## index into search candidates to retain `beam_width` highest-reward sequences
        x = x[inds]
        rewards = rewards[inds]

        ## sample next observation (unless we have reached the end of the planning horizon)
        if t < n_steps - 1:
            x, _ = sample_n(model, x, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## logging
        progress.update({
            'x': list(x.shape),
            'vmin': values.min(), 'vmax': values.max(),
            'qmin': q_vals.min(), 'qmax': q_vals.max(),
            'discount': discount
        })

    progress.stamp()

    ## [ batch_size=256 x (n_context + n_steps) x transition_dim=8 ]
    x = x.view(beam_width, -1, transition_dim)

    ## crop out context transitions
    ## [ batch_size x n_steps x transition_dim ]
    x = x[:, -n_steps:]

    ## return best sequence (15 \times 8)
    argmax = values.argmax()
    best_sequence = x[argmax]

    return best_sequence


@torch.no_grad()
def Best_first_beam_plan(
    model, value_fn,  x,
    n_steps, beam_width, n_expand,
    observation_dim, action_dim,comparison_time,
    discount=0.99, max_context_transitions=None,
    k_obs=None, k_act=None, k_rew=1,
    cdf_obs=None, cdf_act=None, cdf_rew=None,
    verbose=True, previous_actions=None,
):
    '''
        x : tensor[ 1 x input_sequence_length ]
    '''


    # convert max number of transitions to max number of tokens
    transition_dim = observation_dim + action_dim + REWARD_DIM + VALUE_DIM
    max_block = max_context_transitions * transition_dim - 1 if max_context_transitions else None

    ## pass in max numer of tokens to sample function
    sample_kwargs = {
        'max_block': max_block,
        'crop_increment': transition_dim,
    }


    ## construct discount tensors for estimating values
    discounts = discount ** torch.arange(n_steps + 2, device=x.device)
    # rewards = torch.zeros(beam_width, n_steps + 1, device=x.device)

    ## init big_Q
    big_Q = -torch.inf * torch.ones(n_steps+1) # beam_width * 1
    d = {}
    t = 0
    sh_0 = 0 # final score 
    r_0 = torch.zeros(n_steps + 2, device=x.device)
    init_heap = {sh_0 : (x, r_0)}
    big_Q[t] = sh_0
    d[t] = init_heap
    POPS = {}


    ## logging
    # progress = utils.Progress(n_steps) if verbose else utils.Silent()

    # comparison_counter = 0

    ## logging
    progress = utils.Progress(n_steps) if verbose else utils.Silent()
    while n_steps not in d.keys():
        queue_sort = torch.argsort(big_Q)
        t = int(queue_sort[-1])
        top_sh = next(iter(d[t]))
        (seq, rewards) = d[t][top_sh]            # pop out the best seq in t step: d[t]
        del d[t][top_sh]                           # del the best seq in d[t]

        if len(d[t]) == 0:
            big_Q[t] = -torch.inf
        else:
            best_remaining_sh = next(iter(d[t]))
            big_Q[t] = best_remaining_sh


        ## init POPS
        if t not in POPS:
            POPS[t] = 0
        
        ## check beam size beam_width = 128, n_steps = 15
        if POPS[t] >= beam_width or t > n_steps:
            continue
        POPS[t] += 1


        seq = seq.repeat(beam_width * n_expand, 1)
        rewards = rewards.repeat(beam_width * n_expand, 1)
        # actions
        seq , _ = sample_n(model, seq, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)
        # reward
        seq, r_prob = sample_n(model, seq, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_prob)

        ## update rewards tensor
        rewards[:, t] = r_t
        rewards[:, t+1] = V_t
        # print("rewards:", r_t)

        ## estimate scores
        sh = (rewards * discounts).sum(dim=-1) 

        # observation
        if t < n_steps - 1:
            seq, _ = sample_n(model, seq, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        sh, ind = torch.sort(sh, descending=True)
        rewards = rewards[ind]
        seq = seq[ind]

        for i in range(beam_width):
            if t+1 in d.keys():
                if sh[i] not in list(d[t+1].keys()): # clear repeated path
                    d[t+1][sh[i]] = (seq[i], rewards[i])
            else: #we need toq initialize B_t+1
                init_heap = {sh[i]: (seq[i], rewards[i])}
                d[t+1] = init_heap

        best_sh_t1 = next(iter(d[t+1]))
        big_Q[t+1] = best_sh_t1
        ## logging
        progress.update({
            'x': list(x.shape),
            'vmin': sh.min(), 'vmax': sh.max(),
            'vtmin': V_t.min(), 'vtmax': V_t.max(),
            'discount': discount
        })

    progress.stamp()
        ## early stop
        # if comparison_counter >= comparison_time:
        #     best_q = best_q.view(-1, transition_dim)
        #     best_q = best_q[-n_steps:, :]
    
    best_seq = d[n_steps][next(iter(d[n_steps]))][0]
    best_seq = best_seq.view(-1, transition_dim)
    return best_seq


