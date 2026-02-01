import numpy as np

def trajopt_perturb_actions(action_seq, variance, A_min, A_max, samples, seed):
    """Samples perturbed actions for sampling-based TrajOpt.
    Take current "best guess" plan and create K slightly randomized variations of it. 
    explore the neighborhood of current plan to see if small changes yield a better reward

    Args:
        action_seq (array(H, 2)): A single length-H action sequence.
        variance (float): Perturbation variance (sigma in the pseudocode).
        A_min (array(2)): Dimension-wise minima for the box action space.
        A_max (array(2)): Dimension-wise maxima for the box action space.
        samples (int): Number of perturbations to sample.
        seed (int): PRNG seed. Samples for different seeds should be different.
            Samples for the same seed twice should be the same.

    1. Sample Noise: For every time step t and every sample k, generate a noise vector ε_t^k from a Gaussian distribution
    2. Add this noise to current action plan a_t, then "clip" the result so it stays within the robot's physical limits (A_min and A_max)

    
    Returns:
        action_seqs (array(H, samples, 2)): Perturbed actions. Should satisfy
            A_min <= action_seqs[i, j, :] <= A_max elementwise for all i, j.
    """
    action_seqs = np.zeros((action_seq.shape[0], samples, 2))
    # init the random number generator with the seed
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(variance)
    # Generate noise
    # noise[t, k, :] is the 2-D noise vector for time step t and sample index k.
    noise = rng.normal(0, sigma, size=(action_seq.shape[0], samples, 2))
    # Add noise to the action sequence and clip to the action limits
    # replicate the base action at each timestep across K samples and add independent noise for each sample.
    perturbed = action_seq[:, None, :] + noise  # broadcasts action_seq to (H, 1, 2) -> (H, K, 2)
    perturbed = np.clip(perturbed, A_min, A_max)
    # K perturbed versions of the H-step action plan
    return perturbed


def trajopt_rollouts(rmaze, action_seqs):
    """Samples state trajectory rollouts for sampling-based TrajOpt.

    See handout for details. Note: read `rmaze.start` to get the initial state.

    Args:
        rmaze (maze.RobotMaze): Robot-Maze MDP object.
        action_seqs (array(H, K, 2)): K different length-H action sequences.

    Returns:
        states (array(H, K, 3)): State trajectories. For each k in 0..K-1,
            should satisfy states[0, k, :] == rmaze.start, then follow MDP dynamics.
        rewards (array (H, K)): Reward trajectories.
    """
    H, K, _ = action_seqs.shape
    states = np.zeros((H, K, 3))
    rewards = np.zeros((H, K))
    # Set the initial state for all K parallel rollouts
    # rmaze.start is (x, y, theta) - need (K, 3) array
    curr_states = np.tile(rmaze.start, (K, 1))
    for t in range(H):
        states[t] = curr_states
        # Get the actions for this timestep for all K samples
        curr_actions = action_seqs[t]  # shape (K, 2)
        # compute trajectory returns
        rewards[t] = rmaze.reward(curr_states, curr_actions)
        # next states: Roll-out state trajectories.
        curr_states = rmaze.step(curr_states, curr_actions)
         
    return states, rewards

def trajopt_update_opt(action_seqs, reward_seqs):
    """Picks the best action sequence from the perturbed set.

    Args:
        action_seqs (array(H, K, 2)): e.g. from trajopt_perturb_actions.
        reward_seqs (array(H, K)): e.g. from trajopt_rollouts.

    Returns:
        action_seq_opt (array(H, 2)): new guess of the best action sequence.
    """
    # Sum rewards over time to get total reward for each action sequence
    total_rewards = np.sum(reward_seqs, axis=0)  # shape (K,)
    # Find the index of the action sequence with the highest total reward
    best_index = np.argmax(total_rewards)
    # Select the best action sequence
    return action_seqs[:, best_index, :]


def trajopt(rmaze, horizon, variance, samples, iters, seed, plot_ax=None):
    """Implements the Predictive Sampling trajectory optimization algorithm.

    See Algorithm 1 in handout for pseudocode.

    Read `rmaze.start` to get the initial state. Use your own
    trajopt_{perturb_actions, rollouts, update_opt} subroutines to do the work.

    To get proper randomness behavior, you can:
        1) use `seed` to initialize a numpy.random.Generator.
        2) for each iteration, sample a random integer from the Generator, then
           pass it as the seed to trajopt_perturb_actions.

    Action dimension is 2. Initialize the action sequence to all-zeros.

    You are strongly encouraged to implement visualization via the plot_ax
    keyword and RobotMaze.draw_trajs to understand how this algorithm works!

    Args:
        rmaze (maze.RobotMaze): Robot-Maze MDP object.
        horizon (int): Length H of action sequence to plan.
        variance (float): Variance to use in each iteration's noisy sampling.
        samples (int): Number K of random action perturbations to sample.
        iters (int): Number of iterations to perform.
        seed (int): PRNG seed. Result for different seeds should be different.
            Result for the same seed twice should be the same.
        plot_ax (matplotlib Axes object): If set, pass it to RobotMaze.draw
            after each iteration of the main loop to visualize. Not graded, but
            you are strongly encouraged to implement it! See
            trajopt_testcase.py for details.

    Returns:
        aseq (array(H, 2)): Algorithm's chosen action sequence a_1:H.
    """
    return np.zeros((horizon, 2))  # TODO: Implement.



# ------------------ END TRAJOPT / BEGIN TABULAR MODELBASED ------------------



def generate_samples(P, k, seed):
    """Generates k sampled MDP transitions for each MDP (state, action) pair.

    Args:
        P (array(S, A, S)): MDP dynamics, following Assignment 1 conventions.
        k (int): Number of next-state samples to generate per (s, a) pair.
        seed (int): PRNG seed. Samples for different seeds should be different.
            Samples for the same seed twice should be the same.

    Returns:
        s2 (array(S, A, k) of integer): next-state samples, where the integers
            are in the range (0, S - 1).
    """
    S, A, _ = P.shape
    return np.zeros((S, A, k), dtype=int)  # TODO: Implement.
    # HINT: look for a useful method in the class numpy.random.Generator.


def fitmodel(samples):
    """Estimates the MDP transition model using provided samples.

    Use the "Naive" approach based on empirical transition counts from Notes 4.

    Args:
        samples (array(S, A, k) of integer): output of generate_samples.

    Returns:
        Phat (array(S, A, S)): estimate of MDP dynamics P.
    """
    S, A, k = samples.shape
    return np.zeros((S, A, S))  # TODO: Implement.


def analyze_modelbased_error(P, r, gamma, datasets, bellman_iters):
    """Experiment on the impact of estimation error in model-based tabular RL.

    See Task 2 in handout for details.

    For development and learning, you are strongly encouraged to write your own
    "main" function that does the following:

    - Construct an MDP, such a Maze.
    - Use `generate_samples` to construct `datasets`.
    - Call this function and plot the results using matplotlib.

    Args:
        P, r, gamma: True MDP parameters, following conventions of Assignment 1.
        datasets: list of M lists, where each inner list contains N sample
            datasets (arrays) such that `fitmodel(datasets[m][n])` is valid.
        bellman_iters: Number of iterations to use when calling policy
            evaluation and value iteration routines (using your A1 code).

    Returns:
        (E_P, E_star, E_V): tuple of 3x array(M), as described in handout.
    """
    M = len(datasets)
    return (np.zeros(M), np.zeros(M), np.zeros(M))  # TODO: Implement.


if __name__ == "__main__":
    main()
