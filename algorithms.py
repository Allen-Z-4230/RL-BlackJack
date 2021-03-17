import numpy as np
from utils import *

def epsilon_greedy_policy(world, Q, s, epsilon=0.05):
    """Selects actions in an epsilon-greedy manner
    """
    explore = np.random.binomial(1, epsilon)
    if explore:
        return np.random.choice(world.actions)
    else:
        return world.actions[np.argmax(Q[tuple(s)])]

def sarsa_lamdba(lam, env, mode, alpha=0.01, gamma=1, n_ep=1000):
    """Epislon-Greedy SARSA(Lambda) with random starts.

    lam: controls the decay of the eligibility trace
    world: constructor for the environment
    mode: one of ['full', 'hidden']
        full: the dealer's hand is fully observed
        hidden: the dealer only shows one card
    """
    shape0 = 22
    if mode == 'full':
        shape1 = 22
    elif mode == 'hidden':
        shape1 = 11
    actions = ['hit', 'stick']
    n_actions = len(actions)
    Q = np.zeros((shape0, shape1, n_actions))
    for i in range(n_ep):
        E = np.zeros((shape0, shape1, n_actions))
        world = env(mode=mode)
        s = world.start(ret_hand=True)
        a = np.random.choice(actions)
        while s != world.terminal:
            # step through environment
            s_n, r, _ = world.step_agent(s, a)
            a_n = epsilon_greedy_policy(world, Q, s_n)
            a_i = world.actions.index(a)
            a_n_i = world.actions.index(a_n)

            # updates
            delt = r + gamma*Q[tuple(s_n)][a_n_i] - Q[tuple(s)][a_i]
            E[tuple(s)][a_i] += 1  # accumulating trace
            Q += alpha*delt*E
            E *= gamma*lam

            # move to new state
            s = s_n
            a = a_n
    return Q

def gen_episode(env, policy, P, ret_world = False, belief = 'sample', **kwargs):
    """Generates a episode for an episodic world following policy pi, starting with the given state-action pair.

    env: constructor for the world
    policy: array in the shape of (n_states,)
    P: Belief state probabilities.
    belief: one of ['sample', 'max']
        - sample uses the learned distribution to sample the belief space
        - max simply takes the estimate with the maximum probability
    **kwargs: arguments to pass to constructor
    """
    world = env(**kwargs)
    sa_hist = []
    r_hist = []
    hidden = []

    s = world.start(ret_hand=True)
    s0 = s.copy()  # stores separate s0 for updating belief states
    if belief == 'sample':
        h_est = np.random.choice(11, p=softmax(P[tuple(s)]))  # samples the belief state
    elif belief == 'max':
        h_est = np.argmax(P[tuple(s)])  # maximum aperiori estimate of current state

    while s != world.terminal:
        s[1] += h_est  # adjust dealer hand with current best estimate
        s = tuple(s)
        a = policy[s]
        sa_hist.append((s, world.actions.index(a)))
        s_n, r, h = world.step_agent(s, a)
        r_hist.append(r)
        hidden.append(h)
        s = s_n

    if ret_world:
        return world, sa_hist, r_hist, hidden[-1]
    else:
        return s0, sa_hist, r_hist, hidden[-1]

def pomdp_monte_carlo(env, belief, gamma=1, n_ep=1000):
    """Modified Monte Carlo Algorithm for estimating starting probabilities of hidden states & optimal policy.
    """
    shape = 22
    actions = ['hit', 'stick']
    n_actions = len(actions)
    pi = np.random.choice(actions, size=(shape, shape))
    Q = np.zeros((shape, shape, n_actions)) # dealer first card, player sum, hit/stick
    N = np.zeros((shape, shape, n_actions))

    # for updating belief state probabilities
    Np = np.zeros((shape, 11))
    P = np.zeros((shape, 11, 11))

    for i in range(n_ep):
        s0, sa_hist, r_hist, h = gen_episode(env, pi, P, mode='pomdp', belief = belief, ret_world = False)

        # belief state incremental update
        s0 = tuple(s0)
        Np[s0] += 1
        P[s0][h] += (1/Np[s0])*(1 - P[s0][h])

        G = 0
        for t in range(len(sa_hist))[::-1]:
            G = gamma*G + r_hist[t]
            s, a = sa_hist[t]
            s = tuple(s)
            N[s][a] += 1
            Q[s][a] += (1/N[s][a])*(G-Q[s][a])
            pi[s] = actions[np.argmax(Q[s])]
    return pi, Q, N, Np, P
