import numpy as np


"""Assignment-wide conventions:

These notations and arguments mean the same thing throughout the assignment.
Notation: S = number of states, A = number of actions.
The following arguments and/or return values always mean:

    P (array(S, A, S)): Transition dynamics: P[s, a, :] is a S-dimensional
        vector representing the probability distribution over the next state s'
        when starting from state s and taking action a. Note this layout makes
        the expressions look slightly different from the lecture notes math:
        The distribution P(.|s,a) is indexed as P[s, a, :] (or just P[s, a]).
        = P(s'|s,a)

    r (array(S, A)): Reward function: r[s, a] is a scalar representing the
        reward of taking action a from state s.

    gamma (float): The infinite-horizon discount factor, within (0, 1).

    pi (array(S, A)): Policy: pi[s, :] is an A-dimensional vector representing
        the probability distribution over the action a taken at state s. Has
        the same math-vs-code difference as P.
        = pi(a|s)

    V (array(S)): State-value function: current value estimate V(s') for every state starting from state s
"""

"""Task 1: Implement all 4 variants of Bellman operator for tabular MDPs.

The test cases will be randomly generated MDPs. However, you will only receive
binary pass/fail information from Gradescope, so you should not use Gradescope
for development. Instead, you should write your own tests using your test-case
MDPs from Task 2, as well as the Maze MDP tools from maze.py. 
"""

# Computes expected immediate reward plus discounted expected next-state V under pi.
def bellman_V_pi(P, r, gamma, pi, V):
    """Performs one policy Bellman update on the state value function candidate V for policy pi."""
    # immediate reward plus the discounted expected value of the next state s'
    # average over actions a according to policy pi and over next states s' according to transition dynamics P
    expected_values_s_a = ((gamma * (P @ V)) + r)  # shape (S, A)
    weighted_values = pi * expected_values_s_a  # shape (S, A)
    V = np.sum(weighted_values, axis=1)  # shape (S,) - sum over actions to get expected value for each state
    return V

def bellman_Q_pi(P, r, gamma, pi, Q):
    """Performs one policy Bellman update on the state-action value function candidate Q for policy pi."""
# Computes immediate reward plus discounted expected next-state value under policy pi for each state-action pair.
    sum_a = np.sum(pi * Q, axis=1)  # shape (S,) - expected value for each state under policy pi
    expected_next_V = P @ sum_a  # shape (S, A) - expected value of next state s' for each (s, a)
    Q = r + gamma * expected_next_V  # shape (S, A) - immediate reward plus discounted expected value of next state
    return Q

def bellman_V_opt(P, r, gamma, V):
    """Performs one optimal Bellman update on the state value function candidate V."""
# Computes immediate reward plus discounted expected next-state V using greedy (max over actions) update.
    V = np.max(r + gamma * (P @ V), axis=1)  # shape (S,) - max over actions
    return V


def bellman_Q_opt(P, r, gamma, Q):
    """Performs one optimal Bellman update on the state-action value function candidate Q."""
# Computes immediate reward plus discounted expected next-state value where next-state value uses the max over actions.
    max_next_Q = np.max(Q, axis=1)  # shape (S,) - max over actions for each state
    expected_next_V = P @ max_next_Q  # shape (S, A) - expected value of next state s' for each (s, a)
    Q = r + gamma * expected_next_V # shape (S, A) - immediate reward plus discounted expected value of next state
    return Q


"""Task 2: Implement some test MDPs.

For the very simplest test MDPs, you can figure out the optimal value function
manually without the use of an algorithm. Write code to construct them in the
_Vopt functions. Your _Vopt function implementations should directly construct
an array full of hand-computed constants; they should not call your bellman or
value_iteration functions.
"""

def mdp_uniform(S, A):
    """An MDP where actions have no effect.

    Desired output MDP: No matter which action is taken, the reward is 1 and
    the next state is uniformly distributed over the entire state set.

    Args:
        S (int): Number of states.
        A (int): Number of actions.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    # P: Shape (S, A, S)
    # Fill with 1/S so that every next state is equally likely
    P = np.ones((S, A, S)) / S  # Uniform transition probabilities
    r = np.ones((S, A))  # Reward of 1 for all state-action pairs
    return P, r


def mdp_uniform_Vopt(S, A, gamma):
    """Returns the optimal state-value function for mdp_uniform.

    Args:
        S (int): Number of states.
        A (int): Number of actions.
        gamma (float): Discount factor.

    Returns: V (array(S)): optimal value function table.
    V(s) = 1 + gamma * 1 + gamma^2 * 1 + gamma^3 * 1 + ... = 1 / (1 - gamma)
    """
    return np.ones(S) / (1 - gamma)


def mdp_statepick(S):
    """An MDP where the action directly selects the next state.

    Desired output MDP: The number of actions is equal to S. When the i'th
        action is taken, the i'th state is the next state with probability 1. P(s_i|s, a_i) = 1.
        The reward is 1 for taking action 0 in state 0; zero everywhere else.

    Args:
        S (int): Number of states, is also number of actions.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    P = np.zeros((S, S, S))
    r = np.zeros((S, S))
    for s in range(S):
        for a in range(S):
            P[s, a, a] = 1.0  # Taking action a leads to state a with probability 1
    r[0, 0] = 1.0  # Reward of 1 for taking action 0 in state 0
    return P, r

def mdp_statepick_Vopt(S, gamma):
    """Returns the optimal state-value function for mdp_statepick.

    Args:
        S (int): Number of states.
        gamma (float): Discount factor.

    Returns: V (array(S)): optimal value function table.
    """
    V = np.zeros(S)
    V[0] = 1 / (1 - gamma) # Only state 0 has a reward - geometric series of rewards staying in state 0 - V(0) = 1 + gamma * V(0)
    V[1:] = gamma / (1 - gamma)  # All other states: One step delay, then the infinite loop - V(s) = 0 + gamma * V(0)$$
    return V


def mdp_line(S):
    """An MDP where state are connected in a line graph.

    Desired output MDP: States are ordered from left to right numerically.
    There are three actions: 0 = Go left, 1 = don't move, 2 = go right.
    FOR STATES IN THE MIDDLE OF THE LINE:
        The desired outcome happens with probability 0.8. Each of the other two
        possible outcomes happens with probability 0.1.
    FOR STATES AT EITHER END OF THE LINE:
        The probability for "impossible" outcomes (going left from state 0 or
        right from state S - 1) is reassigned to the nearest valid state. For
        example, if we are in state 0 and take action "left", then the next
        state distribtion is:

            P(s' = 0 | s = 0, a = 0) = 0.9, P(s' = 1 | s = 0, a = 0) = 0.1.
    Reward is 1 for *any* action in state 0, and zero otherwise.

    Args:
        S (int): Number of states. Number of actions is fixed at 3.

    Returns: a tuple (P, r), as defined in the assignment-wide conventions.
    """
    A = 3  # Number of actions: 0 = left, 1 = stay, 2 = right
    P = np.zeros((S, A, S))
    r = np.zeros((S, A))
    for s in range(S):
        for a in range(A):
            if a == 0: # Go left
                if s == 0:
                    P[s, a, 0] = 0.9
                    # If there is a state 1, put the 0.1 mass there; otherwise add it to state 0
                    if S > 1:
                        P[s, a, 1] = 0.1
                    else:
                        P[s, a, 0] += 0.1
                else:
                    P[s, a, s - 1] = 0.8
                    P[s, a, s] = 0.1
                    if s + 1 < S:
                        P[s, a, s + 1] = 0.1
                    else:
                        P[s, a, s] += 0.1
            elif a == 1: # Stay
                P[s, a, s] = 0.8
                if s - 1 >= 0:
                    P[s, a, s - 1] = 0.1
                else:
                    P[s, a, s] += 0.1
                if s + 1 < S:
                    P[s, a, s + 1] = 0.1
                else:
                    P[s, a, s] += 0.1
            elif a == 2: # Go right
                if s == S - 1:
                    P[s, a, S - 1] = 0.9
                    # If there is a state S-2 (i.e. S>1) put the 0.1 mass there; otherwise add it to state S-1
                    if S > 1:
                        P[s, a, S - 2] = 0.1
                    else:
                        P[s, a, S - 1] += 0.1
                else:
                    P[s, a, s + 1] = 0.8
                    P[s, a, s] = 0.1
                    if s - 1 >= 0:
                        P[s, a, s - 1] = 0.1
                    else:
                        P[s, a, s] += 0.1
    for a in range(A):
        r[0, a] = 1.0  # Reward of 1 for any action in state 0
    return P, r

"""Task 3: Implement all variants of (Q-)Value Iteration and Policy Evaluation.

Use your Bellman operator implementations from Task 1.
Arguments P, r, gamma, and pi are interpreted as in Task 1.
Argument `iters` controls the number of iterations to perform.
NOTE: The initial guess should always be zero, otherwise tests will fail.
"""

def value_iteration_V(P, r, gamma, iters):
    """Performs `iters` steps of value iteration."""
    S = P.shape[0]
    V = np.zeros(S)
    for _ in range(iters):
        V = bellman_V_opt(P, r, gamma, V)
    return V


def policy_evaluation_V(P, r, gamma, pi, iters):
    """Performs `iters` steps of policy evaluation for policy pi."""
    S = P.shape[0]
    V = np.zeros(S)
    for _ in range(iters):
        V = bellman_V_pi(P, r, gamma, pi, V)
    return V

def value_iteration_Q(P, r, gamma, iters):
    """Performs `iters` steps of Q-value iteration."""
    Q = np.zeros(r.shape)
    for _ in range(iters):
        Q = bellman_Q_opt(P, r, gamma, Q)
    return Q


def policy_evaluation_Q(P, r, gamma, pi, iters):
    """Performs `iters` steps of Q- policy evaluation for policy pi."""
    Q = np.zeros(r.shape)
    for _ in range(iters):
        Q = bellman_Q_pi(P, r, gamma, pi, Q)
    return Q

"""Task 4: Implement Policy Iteration.

The Policy Iteration code should use your policy_evaluation_Q as a subroutine.
For each "policy evaluation" setup, run it for a fixed amount of iterations,
following the argument `PE_iters`. The initial guess should take action 0 from
all states.

PI should terminate whenever the policy does not change after an
evaluation/improvement cycle. The return value should be a policy in the format
of `pi` from the assignment-wide conventions.

PI always generates a greedy policy, so each action distribution (row of pi)
should be "one-hot", i.e. all zeros except for a single one.

Q: entry Q[s, a] is the estimated value of taking action a in state s.
pi: entry pi[s, a] is the probability of taking action a in state s.
"""
def policy_iteration(P, r, gamma, PE_iters):
    """Performs policy iteration, stopping when the policy does not change."""
    S, A = r.shape
    pi = np.zeros((S, A))
    pi[:, 0] = 1.0  # Initial policy: always take action 0
    while True:
        Q = policy_evaluation_Q(P, r, gamma, pi, PE_iters) # shape (S, A) - returns a Q estimate for policy p
        new_pi = np.zeros((S, A))
        best_actions = np.argmax(Q, axis=1) # shape (S,) - scans each row (each state) and returns the column index of the maximum value in that row
        for s in range(S):
            new_pi[s, best_actions[s]] = 1.0 # Greedy policy improvement: take the best action with probability 1
        if np.array_equal(new_pi, pi):
            break
        pi = new_pi
    return pi

