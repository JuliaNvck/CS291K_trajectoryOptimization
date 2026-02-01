import numpy as np
import mdpsolve_student

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

    # for each iteration, sample a random integer from the Generator, then pass it as the seed to trajopt_perturb_actions.
    # Action dimension is 2. Initialize the action sequence to all-zeros.
    action_seq = np.zeros((horizon, 2))
    # initialize a numpy.random.Generator.
    rng = np.random.default_rng(seed)
    for it in range(iters):
        # sample a random integer from the Generator
        iter_seed = rng.integers(0, 1e9)
        # 1. Perturb Actions
        action_seqs = trajopt_perturb_actions(action_seq, variance, rmaze.A_min, rmaze.A_max, samples, iter_seed)
        # 2. Rollouts
        states, rewards = trajopt_rollouts(rmaze, action_seqs)
        # 3. Update Optimal Action Sequence
        action_seq = trajopt_update_opt(action_seqs, rewards)
        # Visualization
        if plot_ax is not None:
            # We need to know which index was 'best' to draw it in red.
            # Re-calculating the index here is cheap and keeps the API clean.
            total_rewards = np.sum(rewards, axis=0)
            best_idx = np.argmax(total_rewards)
            
            # Extract the single best state trajectory (H, 3)
            best_state_traj = states[:, best_idx, :]
            
            # Pass (axis, best_one, all_others)
            rmaze.draw_trajs(plot_ax, best_state_traj, states)

    return action_seq



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
    rng = np.random.default_rng(seed)
    samples = np.zeros((S, A, k), dtype=int)
    for s in range(S):
        for a in range(A):
            samples[s, a] = rng.choice(S, size=k, p=P[s, a]) # sample k next states according to discrete distribution P[s, a]
    return samples


def fitmodel(samples):
    """Estimates the MDP transition model using provided samples.

    Use the "Naive" approach based on empirical transition counts from Notes 4.

    Args:
        samples (array(S, A, k) of integer): output of generate_samples.

    Returns:
        Phat (array(S, A, S)): estimate of MDP dynamics P.
    """
    S, A, k = samples.shape
    Phat = np.zeros((S, A, S))
    for s in range(S):
        for a in range(A):
            for next_s in samples[s, a]:
                Phat[s, a, next_s] += 1
            Phat[s, a] /= k  # normalize to get probabilities
    return Phat


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
    def extract_greedy_policy(Q):
        S, A = Q.shape
        pi = np.zeros((S, A))
        best_actions = np.argmax(Q, axis=1)
        pi[np.arange(S), best_actions] = 1.0
        return pi
    M = len(datasets)
    N = len(datasets[0])  # Number of samples per dataset size
    E_P = np.zeros(M)
    E_star = np.zeros(M)
    E_V = np.zeros(M)
    # Compute optimal value under true dynamics (same for all m, n)
    V_star = mdpsolve_student.value_iteration_V(P, r, gamma, bellman_iters)
    # for each dataset
    for m in range(M):
        # Compute all three errors for each (m, n) pair and average over N samples for each m
        # Accumulate errors across N samples
        ep_sum = 0.0
        estar_sum = 0.0
        ev_sum = 0.0
        for n in range(N):
            # Estimate Phat from Dm,n using fitmodel
            phat = fitmodel(datasets[m][n])
            # compute policy pi_hat using value iteration: Compute Q-table
            Q = mdpsolve_student.value_iteration_Q(phat, r, gamma, bellman_iters)
            # Extract greedy policy: for each state, take the action with highest Q-value
            pi_hat = extract_greedy_policy(Q)
            # Find V πhat, the value of πhat under the real dynamics P , using your implementation of Policy Evaluation 
            V_pihat = mdpsolve_student.policy_evaluation_V(P, r, gamma, pi_hat, bellman_iters)
            # Find V πhat, Phat , the value of πhat under the incorrect, estimated dynamics Phat
            # read this directly from the QVI solution 
            V_pihat_phat = np.max(Q, axis=1)
            # compute the worst-case estimation error of a single (state, action, next-state) transition probability across the entire MDP.
            error_P = np.max(np.abs(P - phat))
            ep_sum += error_P
            # compute the worst-case suboptimality of πhat across S.
            error_star = np.max(V_star - V_pihat)
            estar_sum += error_star
            # compute how the model Phat overestimates the value of πhat compared to its value under the true dynamics
            error_value = np.max(V_pihat_phat - V_pihat)
            ev_sum += error_value
        # average over N samples
        E_P[m] = ep_sum / N
        E_star[m] = estar_sum / N
        E_V[m] = ev_sum / N
    return (E_P, E_star, E_V)


# if __name__ == "__main__":
#     main()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # If your MDP functions are in a different file, import them. 
    # Otherwise, ensure mdp_line, mdp_statepick etc. are available here.
    # from mdpsolve_student import mdp_line 

    # 1. Setup the Experiment
    print("Setting up experiment...")
    S = 10
    # Create a test MDP (Line world is good for this)
    # Ensure you have mdp_line available or copy it from your previous code
    P_true, r_true = mdpsolve_student.mdp_line(S) 
    gamma = 0.9
    
    # Define the range of sample sizes (k) to test
    # e.g., 1 sample per (s,a), then 2, 5, 10, ... up to 500
    k_values = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    
    # Define how many random trials (N) to run for each k to get a smooth average
    n_trials = 20
    
    # 2. Generate the Datasets
    # structure: datasets[m][n] is the dataset for k=k_values[m] and seed=n
    datasets = []
    
    print(f"Generating datasets for k values: {k_values} with {n_trials} trials each...")
    
    rng = np.random.default_rng(0) # Master seed
    
    for k in k_values:
        datasets_for_k = []
        for n in range(n_trials):
            # Generate a unique seed for this specific trial
            trial_seed = rng.integers(0, 1000000)
            
            # Generate samples using your function
            # P_true shape is (S, A, S) from mdp_line
            data = generate_samples(P_true, k, trial_seed)
            datasets_for_k.append(data)
        datasets.append(datasets_for_k)

    # 3. Run Analysis
    print("Running analysis (this might take a moment)...")
    # Use a sufficient number of iterations for VI/PE to converge
    bellman_iters = 100 
    
    # Call your analysis function
    E_P, E_star, E_V = analyze_modelbased_error(P_true, r_true, gamma, datasets, bellman_iters)

    # 4. Plot Results
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    
    # Plot all three errors on a log-log scale because error usually decays polynomially (1/sqrt(k))
    plt.loglog(k_values, E_P, label=r"Model Error $E^P$", marker='o')
    plt.loglog(k_values, E_star, label=r"Suboptimality $E^*$", marker='s')
    plt.loglog(k_values, E_V, label=r"Value Error $E^V$", marker='^')
    
    plt.xlabel("Number of Samples $k$ per (s, a) pair")
    plt.ylabel("Error")
    plt.title("Model-Based RL Error vs. Sample Size (Log-Log Plot)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    output_filename = "model_based_error_analysis.png"
    plt.savefig(output_filename)
    print(f"Done! Plot saved to {output_filename}")
    plt.show()