import numpy as np
import torch
import pdb
import heapq
from .. import utils
from .sampling import sample_n, sort_2d, sample,sample_n_topp

REWARD_DIM = VALUE_DIM = 1




def reconstruct(indices, subslice=(None, None)):

    if torch.is_tensor(indices):
        indices = indices.detach().cpu().numpy()

    ## enforce batch mode
    if indices.ndim == 1:
        indices = indices[None]

    if indices.min() < 0 or indices.max() >= self.N:
        print(f'[ utils/discretization ] indices out of range: ({indices.min()}, {indices.max()}) | N: {self.N}')
        indices = np.clip(indices, 0, self.N - 1)

    start, end = subslice
    thresholds = self.thresholds[:, start:end]

    left = np.take_along_axis(thresholds, indices, axis=0)
    right = np.take_along_axis(thresholds, indices + 1, axis=0)
    recon = (left + right) / 2.
    return recon

@torch.no_grad()
def beam_search(model, x, n_steps, beam_width=512, goal=None, **sample_kwargs):
    batch_size = len(x)

    prefix_i = torch.arange(len(x), dtype=torch.long, device=x.device)
    cumulative_logp = torch.zeros(batch_size, 1, device=x.device)

    for t in range(n_steps):

        if goal is not None:
            goal_rep = goal.repeat(len(x), 1)
            logp = get_logp(model, x, goal=goal_rep, **sample_kwargs)
        else:
            logp = get_logp(model, x, **sample_kwargs)

        candidate_logp = cumulative_logp + logp
        sorted_logp, sorted_i, sorted_j = sort_2d(candidate_logp)

        n_candidates = (candidate_logp > -np.inf).sum().item()
        n_retain = min(n_candidates, beam_width)
        cumulative_logp = sorted_logp[:n_retain].unsqueeze(-1)

        sorted_i = sorted_i[:n_retain]
        sorted_j = sorted_j[:n_retain].unsqueeze(-1)

        x = torch.cat([x[sorted_i], sorted_j], dim=-1)
        prefix_i = prefix_i[sorted_i]

    x = x[0]
    return x, cumulative_logp.squeeze()



@torch.no_grad()
def beam_stop(big_Q, n_steps):
    for q in big_Q:
        if q.shape[0] < n_steps - 1:
            return False
    return  True

def heuristic_f(seq):
    pass

class HeapItem:
    # p = ()
    def __init__(self, p):
        self.p = p

    def __lt__(self, other): 
        return self.p < other.p
    
    def __len__(self):
        return len(self.p)
    
    def get(self):
        return self.p
    
@torch.no_grad()
def A_star_beam_plan(
    model, value_fn, x,
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

    ## init Q, big_Q
    Q = []
    big_Q = [] # beam_width * 1
    heapq.heapify(Q)
    heapq.heapify(big_Q)
    
    heapq.heappush(Q, (HeapItem(torch.tensor(0)), 0, HeapItem(x), HeapItem(torch.zeros(n_steps + 2, device=x.device)))) # 0-15
    # (scores, times of decision), sequences, rewards

    heapq.heappush(big_Q, (HeapItem(torch.tensor(0)), Q) )
    # Highest scores,  sequences

    POPS = {}


    ## logging
    # progress = utils.Progress(n_steps) if verbose else utils.Silent()

    best_sh = -1e10
    best_q = []

    
    comparison_counter = 0

    while len(Q) != 0 and not beam_stop(big_Q, n_steps):
        sh, t, seq, r = heapq.heappop(Q) 
        sh = -sh.get()
        seq = seq.get()
        r = r.get().repeat(n_expand ,1)

        # print("length of seq:", len(seq))
        # print("times of decision:", t)
        # print("length of Q:", len(Q))

        ## init POPS
        if t not in POPS:
            POPS[t] = 0
        
        ## check beam size beam_width = 128, n_steps = 15
        if POPS[t] >= beam_width or t > n_steps:
            continue
        POPS[t] += 1

        ## compare
        if t == n_steps:
            comparison_counter += 1
            if sh > best_sh:
                best_q = seq
                best_sh = sh
        

        ## push new seq 
        if len(seq) >= len(best_q):
            seq = seq.repeat(n_expand, 1)
        else:
            seq = seq.repeat(1, 1)
        
        # actions
        seq , _ = sample_n(model, seq, action_dim, topk=k_act, cdf=cdf_act, **sample_kwargs)
        # reward
        seq, r_prob = sample_n(model, seq, REWARD_DIM + VALUE_DIM, topk=k_rew, cdf=cdf_rew, **sample_kwargs)
        # observation
        if t < n_steps - 1:
            seq, _ = sample_n(model, seq, observation_dim, topk=k_obs, cdf=cdf_obs, **sample_kwargs)

        ## optionally, use a percentile or mean of the reward and
        ## value distributions instead of sampled tokens
        r_t, V_t = value_fn(r_prob)

        ## update rewards tensor
        r[:, t] = r_t
        r[:, t+1] = V_t
        # print("rewards:", r_t)

        ## estimate scores
        sh = (r * discounts).sum(dim=-1) 

        # + heuristic_f(seq) Q-learning (offline) (IQL CQL)


        #divide sh
        for sh_i, rew_i, seq_i in zip(torch.chunk(sh, n_expand, dim=0), 
                                      torch.chunk(r, n_expand, dim = 0),
                                      torch.chunk(seq, n_expand, dim = 0)):

            heapq.heappush(Q, (HeapItem(-sh_i), t+1, HeapItem(seq_i[0]), HeapItem(rew_i[0])))
                ## Q -> (k, )
        Q = heapq.nsmallest(beam_width, Q)

        ## early stop
        if comparison_counter >= comparison_time:
            best_q = best_q.view(-1, transition_dim)
            best_q = best_q[-n_steps:, :]

            return best_q
        
        # else:
        #     return None

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





