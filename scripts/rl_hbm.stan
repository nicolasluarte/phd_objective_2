// Stan model for Hierarchical Q-Learning (alpha, tau)
// With Groups, Phases, Daily Q-Resets
// --- MODIFIED for reduce_sum within-chain parallelization ---

functions {
  // Log-softmax function (numerically stable) - NO CHANGE
  vector log_softmax_stable(vector Q, real tau) {
    if (tau <= 0) {
      reject("Temperature tau must be positive, but is ", tau);
    }
    vector[num_elements(Q)] scaled_Q = Q / tau;
    return scaled_Q - log_sum_exp(scaled_Q);
  }

  // --- NEW: Slicing function for reduce_sum ---
  // Calculates the sum of log-likelihoods for a slice of subjects
  real partial_log_lik(
    array[] int subj_indices_slice, // Indices of subjects in this slice (e.g., {5,6,7,8})
    int start,                     // Start index within subj_indices_slice (relative to slice)
    int end,                       // End index within subj_indices_slice (relative to slice)

    // --- SHARED ARGUMENTS (passed from reduce_sum call) ---
    int n_choices,                 // Number of choices
    // Full data arrays (length N - total trials)
    array[] int phase_id,
    array[] int day_id,
    array[] int actions,
    array[] real rewards,
    // Subject boundary indices (length P)
    array[ ] int subj_start_idx, // Stan requires array[] for slicing func args if array
    array[ ] int subj_end_idx,
    // Parameter matrices (size P x 2)
    matrix alpha,
    matrix tau
  ) {
    real slice_total_log_lik = 0.0; // Accumulator for this slice
    vector[n_choices] Q;          // Q-values for the subject being processed in the inner loop

    // Loop through the subjects assigned to this slice
    for (i in start:end) {
      int p = subj_indices_slice[i]; // Get the actual subject ID (1 to P)

      // Get trial boundaries for this subject p
      int T_start = subj_start_idx[p];
      int T_end = subj_end_idx[p];

      // Check if subject has any trials (should always be true if P>0)
      if (T_start > T_end) continue; // Skip if subject has no trials somehow

      // Initialize Q for this subject
      Q = rep_vector(0.5, n_choices);

      // Loop ONLY through trials for subject p
      for (n in T_start:T_end) {
        int ph = phase_id[n]; // Phase for this trial

        // --- Q-Reset Logic for Day Change ---
        // Check if day changed compared to previous trial *for the same subject*
        // Requires careful indexing for the very first trial of the subject (n == T_start)
        //if (n > T_start) { // Only check previous trial if not the first one
          // if (day_id[n] != day_id[n - 1]) {
            //  Q = rep_vector(0.5, n_choices); // Reset if day changed
          // }
        //} // Q was already reset at T_start

        // Get parameters for this subject & phase
        real current_alpha = alpha[p, ph];
        real current_tau = tau[p, ph];

        // Calculate log-softmax
        vector[n_choices] log_softmax_p = log_softmax_stable(Q, current_tau);

        // Accumulate log likelihood for this trial
        slice_total_log_lik += log_softmax_p[actions[n]];

        // Update Q-value
        real PE = rewards[n] - Q[actions[n]];
        Q[actions[n]] = Q[actions[n]] + current_alpha * PE;

        // Optional clamp if used before
        // Q[actions[n]] = clamp(Q[actions[n]], 0.0, 1.0);
      } // End loop over trials n for subject p
    } // End loop over subjects i in slice

    return slice_total_log_lik; // Return sum of log-likelihoods for this slice
  } // End partial_log_lik function
} // End functions block

data {
  // Data sizes (same as before)
  int<lower=1> N; int<lower=1> P; int<lower=1> n_choices; int<lower=1> n_groups;

  // Trial-level data arrays (length N)
  array[N] int<lower=1, upper=P> subj_id;
  array[N] int<lower=1, upper=2> phase_id;
  array[N] int<lower=1> day_id;
  array[N] int<lower=1, upper=n_choices> actions;
  array[N] real rewards;

  // Subject-level data arrays (length P)
  array[P] int<lower=1, upper=n_groups> group_id;
  // --- NEW: Subject trial boundaries ---
  array[P] int<lower=1, upper=N> subj_start_idx; // Start index (in N) for each subject p
  array[P] int<lower=1, upper=N> subj_end_idx;   // End index (in N) for each subject p
}

parameters { // NO CHANGE needed here
  matrix[n_groups, 2] mu_pr_alpha; matrix[n_groups, 2] mu_log_tau;
  real<lower=0> sigma_pr_alpha; real<lower=0> sigma_log_tau;
  matrix[P, 2] z_pr_alpha; matrix[P, 2] z_log_tau;
}

transformed parameters { // NO CHANGE needed here
  matrix[P, 2] pr_alpha; matrix[P, 2] log_tau;
  for (p in 1:P) {
    int g = group_id[p];
    for (ph in 1:2) {
      pr_alpha[p, ph] = mu_pr_alpha[g, ph] + sigma_pr_alpha * z_pr_alpha[p, ph];
      log_tau[p, ph] = mu_log_tau[g, ph] + sigma_log_tau * z_log_tau[p, ph];
    }
  }
  matrix<lower=0, upper=1>[P, 2] alpha; matrix<lower=0>[P, 2] tau;
  for (p in 1:P) {
    for (ph in 1:2) {
      alpha[p, ph] = Phi(pr_alpha[p, ph]);
      tau[p, ph] = exp(log_tau[p, ph]);
    }
  }
}

model {
  // Priors (NO CHANGE needed here)
  to_vector(mu_pr_alpha) ~ normal(0, 1); to_vector(mu_log_tau) ~ normal(0, 1.5);
  sigma_pr_alpha ~ normal(0, 1); sigma_log_tau ~ normal(0, 1);
  to_vector(z_pr_alpha) ~ normal(0, 1); to_vector(z_log_tau) ~ normal(0, 1);

  // --- Likelihood using reduce_sum ---
  // Define grainsize (how many subjects per chunk sent to a thread)
  // Start with 1, might tune later based on performance/cores.
  int grainsize = 1;

  // Array of subject indices (1, 2, ..., P) to slice over
  array[P] int subj_indices;
  for (p in 1:P) {
    subj_indices[p] = p;
  }

  // Call reduce_sum to parallelize the likelihood calculation across subjects
  target += reduce_sum(
    partial_log_lik,     // The slicing function defined above
    subj_indices,        // The array we are slicing (indices of subjects)
    grainsize,           // The grainsize
    // --- SHARED ARGUMENTS (passed to every call of partial_log_lik) ---
    // Need to pass all data arrays and parameter matrices needed by the function
    n_choices,
    phase_id, day_id, actions, rewards, // Trial-level data (full arrays)
    subj_start_idx, subj_end_idx,       // Subject boundaries
    alpha, tau                          // Individual parameters [P, 2]
  );
}

generated quantities {
  // Keep this simple for now, or replicate reduce_sum logic if log_lik needed
  // For speed, maybe calculate only group means/differences here.
  // If log_lik is essential, you *could* use reduce_sum here too,
  // but it would need modification to return the vector[N] log_lik.
  // Simpler approach: calculate log_lik post-hoc in R if needed, using draws.

  // Calculations of group means/differences (same as before)
  matrix[n_groups, 2] group_mean_alpha; matrix[n_groups, 2] group_mean_tau;
  // ... (rest of difference calculations) ...
  for (g in 1:n_groups) {
    for (ph in 1:2) {
      group_mean_alpha[g, ph] = Phi(mu_pr_alpha[g, ph]);
      group_mean_tau[g, ph] = exp(mu_log_tau[g, ph]);
    }
  }
  vector[n_groups] change_alpha = group_mean_alpha[, 2] - group_mean_alpha[, 1];
  vector[n_groups] change_tau = group_mean_tau[, 2] - group_mean_tau[, 1];
  real diff_in_diff_alpha = 0.0; real diff_in_diff_tau = 0.0;
  if (n_groups == 2) {
    diff_in_diff_alpha = change_alpha[2] - change_alpha[1];
    diff_in_diff_tau = change_tau[2] - change_tau[1];
  }
}
