// Stan model for Hierarchical Q-Learning (alpha, tau)
// With Groups (A vs B), Phases (1 vs 2), and Daily Q-Resets

functions {
  vector log_softmax_stable(vector Q, real tau) {
    if (tau <= 0) {
      reject("Temperature tau must be positive, but is ", tau);
    }
    vector[num_elements(Q)] scaled_Q = Q / tau;
    return scaled_Q - log_sum_exp(scaled_Q);
  }
}

data {
  // Data sizes
  int<lower=1> N;          // Total number of trials across all subjects & phases & days
  int<lower=1> P;          // Number of subjects
  int<lower=1> n_choices;  // Number of distinct actions
  int<lower=1> n_groups;   // Number of treatment groups

  // Data arrays
  array[N] int<lower=1, upper=P> subj_id;  // Subject ID for each trial n (1 to P)
  array[N] int<lower=1, upper=2> phase_id; // Phase ID for each trial n (1 or 2)
  // --- NEW: Day Information ---
  array[N] int<lower=1> day_id;      // Day ID/number for each trial n (e.g., 1, 2, 3...)

  array[P] int<lower=1, upper=n_groups> group_id; // Group ID (1 or 2) for each subject p
  array[N] int<lower=1, upper=n_choices> actions; // Actions taken
  array[N] real rewards;     // Rewards received
}

parameters {
  // Group-level parameters (indexed by [group, phase])
  matrix[n_groups, 2] mu_pr_alpha; // Mean probit(alpha)
  matrix[n_groups, 2] mu_log_tau;  // Mean log(tau)
  real<lower=0> sigma_pr_alpha; // Common SD probit(alpha)
  real<lower=0> sigma_log_tau;  // Common SD log(tau)

  // Individual-level raw deviations (indexed by [subject, phase])
  matrix[P, 2] z_pr_alpha;
  matrix[P, 2] z_log_tau;
}

transformed parameters {
  // Individual parameters (transformed scale), indexed by [subject, phase]
  matrix[P, 2] pr_alpha;
  matrix[P, 2] log_tau;
  for (p in 1:P) {
    int g = group_id[p];
    for (ph in 1:2) {
      pr_alpha[p, ph] = mu_pr_alpha[g, ph] + sigma_pr_alpha * z_pr_alpha[p, ph];
      log_tau[p, ph] = mu_log_tau[g, ph] + sigma_log_tau * z_log_tau[p, ph];
    }
  }

  // Transform back to original scale, indexed by [subject, phase]
  matrix<lower=0, upper=1>[P, 2] alpha;
  matrix<lower=0>[P, 2] tau;
  for (p in 1:P) {
    for (ph in 1:2) {
      alpha[p, ph] = Phi(pr_alpha[p, ph]);
      tau[p, ph] = exp(log_tau[p, ph]);
    }
  }
}

model {
  // Priors (same as before)
  to_vector(mu_pr_alpha) ~ normal(0, 1);
  to_vector(mu_log_tau) ~ normal(0, 1.5);
  sigma_pr_alpha ~ normal(0, 1);
  sigma_log_tau ~ normal(0, 1);
  to_vector(z_pr_alpha) ~ normal(0, 1);
  to_vector(z_log_tau) ~ normal(0, 1);

  // Likelihood
  {
    vector[n_choices] Q;
    for (n in 1:N) {
      int p = subj_id[n];
      int ph = phase_id[n];

      // --- MODIFIED: Reset Q for new subject OR new day ---
      // Assumes data is ordered: subject -> day -> trial
      // Assumes day_id uniquely identifies the day for that subject (can restart for new subj)
      if (n == 1 || subj_id[n] != subj_id[n - 1] || day_id[n] != day_id[n - 1]) {
        Q = rep_vector(0.5, n_choices); // Reset Q at start of subject or day
      }

      // Select correct parameters for the current subject/phase
      real current_alpha = alpha[p, ph];
      real current_tau = tau[p, ph];

      // Calculate log-softmax using phase-specific tau
      vector[n_choices] log_softmax_p = log_softmax_stable(Q, current_tau);

      // Increment target log probability
      target += log_softmax_p[actions[n]];

      // Update Q-value using phase-specific alpha
      real PE = rewards[n] - Q[actions[n]];
      Q[actions[n]] = Q[actions[n]] + current_alpha * PE;
    }
  }
}

generated quantities {
  vector[N] log_lik;
  {
    vector[n_choices] Q_gen;
    for (n in 1:N) {
      int p = subj_id[n];
      int ph = phase_id[n];

      // --- MODIFIED: Reset Q_gen logic (must match model block exactly) ---
      if (n == 1 || subj_id[n] != subj_id[n - 1] || day_id[n] != day_id[n - 1]) {
        Q_gen = rep_vector(0.5, n_choices);
      }

      real current_alpha_gen = alpha[p, ph];
      real current_tau_gen = tau[p, ph];
      vector[n_choices] log_softmax_p_gen = log_softmax_stable(Q_gen, current_tau_gen);
      log_lik[n] = log_softmax_p_gen[actions[n]];
      real PE_gen = rewards[n] - Q_gen[actions[n]];
      Q_gen[actions[n]] = Q_gen[actions[n]] + current_alpha_gen * PE_gen;
    }
  }

  // --- Calculations of group means/differences (same as before) ---
  matrix[n_groups, 2] group_mean_alpha;
  matrix[n_groups, 2] group_mean_tau;
  // ... (rest of generated quantities for differences etc. remains the same) ...
  for (g in 1:n_groups) {
    for (ph in 1:2) {
      group_mean_alpha[g, ph] = Phi(mu_pr_alpha[g, ph]);
      group_mean_tau[g, ph] = exp(mu_log_tau[g, ph]);
    }
  }
  vector[n_groups] change_alpha = group_mean_alpha[, 2] - group_mean_alpha[, 1];
  vector[n_groups] change_tau = group_mean_tau[, 2] - group_mean_tau[, 1];
  real diff_in_diff_alpha = 0.0;
  real diff_in_diff_tau = 0.0;
  if (n_groups == 2) {
    diff_in_diff_alpha = change_alpha[2] - change_alpha[1];
    diff_in_diff_tau = change_tau[2] - change_tau[1];
  }
}
