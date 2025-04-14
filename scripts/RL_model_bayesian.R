pacman::p_load(
   cmdstanr,
   tidyverse,
   posterior,
   furrr,
   readr,
   bayesplot,
   bayestestR
)

setwd(this.path::here())
# see RL_model script if you need to modify data
individual_choice_data <- read_csv("../datasets/individual_choice_data.csv") %>% 
    mutate(
        exp_group = as.numeric(if_else(exp_group=="control",1,2)),
        exp_phase = as.numeric(if_else(exp_phase=="basal", 1, 2)),
        ID = as.numeric(as.factor(ID)),
        rel_date = rel_date+1
    ) %>% 
    ungroup() %>% 
    mutate(overall_trial_index = 1:n())

subject_boundaries <- individual_choice_data %>% 
    group_by(ID) %>% 
    summarise(
        subj_start_idx = min(overall_trial_index),
        subj_end_idx = max(overall_trial_index),
        .groups = "drop"
    ) %>% 
    arrange(ID)


stan_data_list <- list(
    N = nrow(individual_choice_data),
    P = length(unique(individual_choice_data$ID)),
    n_choices = 2,
    n_groups = length(unique(individual_choice_data$exp_group)),
    group_id = individual_choice_data %>% distinct(ID,exp_group) %>% pull(exp_group),  # Maps subject p -> group g
    subj_id = individual_choice_data$ID,     # Maps trial n -> subject p
    phase_id = individual_choice_data$exp_phase,   # Maps trial n -> phase ph
    day_id = individual_choice_data$rel_date, # <<< NEW: Maps trial n -> unique day number for subject
    actions = individual_choice_data$actions,     # Action for trial n
    rewards = individual_choice_data$rewards,    # Reward for trial n
    subj_start_idx = subject_boundaries$subj_start_idx,
    subj_end_idx = subject_boundaries$subj_end_idx
)

stan_file <- "rl_hbm.stan" # Use the new Stan file name
mod <- cmdstan_model(stan_file,
                     compile = TRUE)

bayesian_fit <- mod$sample(
    data = stan_data_list,
    init = 0,
    seed = 1234567,
    chains = 4,
    threads_per_chain = 3,
    parallel_chains = 4,
    iter_warmup = 1000, # Consider increasing warmup
    iter_sampling = 2000, # Consider increasing sampling
    refresh = 10,
    adapt_delta = 0.95, # Higher value often needed
    max_treedepth = 12 # Might need increasing
)
bayesian_fit$save_object(file = "../datasets/bayesian_fit_2025_04_12.rds")


bayesian_fit$draws(variables = c("group_mean_alpha", "group_mean_tau"), format="df") %>% 
    pivot_longer(
        cols = everything(),
        names_to = "variable",
        values_to = "value"
    ) %>% 
    filter(!variable %in% c(".chain", ".iteration", ".draw"),
           grepl("tau", variable)) %>% 
    ggplot(aes(
        value, fill=variable
    )) +
    geom_density(alpha=0.5) +
    xlim(c(0, 1))

did_draws <- bayesian_fit$draws(variables = c("diff_in_diff_alpha", "diff_in_diff_tau"),
                       format = "df")

draws_did_alpha <- bayesian_fit$draws(variables = "diff_in_diff_alpha", format = "matrix")
bf_did_alpha <- bayesfactor_parameters(draws_did_alpha, null = 0)
bf_did_alpha

hdi_results <- hdi(did_draws, .width = 0.95)
hdi_results

desc_post <- describe_posterior(did_draws, ci = 0.95, test = c("p_direction", "rope", "bayesfactor"))
desc_post

# Calculate median and 95% Credible Interval (Equal-Tailed Interval)
did_summary <- summarise_draws(did_draws,
                               ~quantile(.x, probs = c(0.025, 0.5, 0.975)))
did_summary

individual_summaries <- bayesian_fit$summary(
    variables = c("alpha", "tau"), # Get summaries for all elements of alpha and tau
    "mean",
    "median",
    q5 = ~quantile(.x, probs = 0.05), # 90% CI lower bound / 95% one-sided lower
    q95 = ~quantile(.x, probs = 0.95),# 90% CI upper bound / 95% one-sided upper
    q2.5 = ~quantile(.x, probs = 0.025), # 95% CI lower bound
    q97.5 = ~quantile(.x, probs = 0.975) # 95% CI upper bound
    # You can add other summaries like "sd", "rhat", "ess_bulk" if needed
)
individual_summaries

pd_alpha <- mean(did_draws$diff_in_diff_alpha > 0) # Probability that difference > 0
pd_tau <- mean(did_draws$diff_in_diff_tau > 0)
pd_alpha
pd_tau


fit <-  bayesian_fit$draws(format = "matrix")

# Plot posterior density for the difference-in-difference in alpha
plot_did_alpha <- mcmc_hist(fit, pars = "diff_in_diff_alpha", binwidth = 0.02) + # Adjust binwidth as needed
    vline_0(colour = "red", linetype = "dashed") +
    labs(
        title = "Posterior for Difference-in-Difference (Alpha)",
        subtitle = "Change(Group B) - Change(Group A)",
        x = "Difference in Alpha Change"
    )
print(plot_did_alpha)

# Repeat for tau
plot_did_tau <- mcmc_hist(fit, pars = "diff_in_diff_tau", binwidth = 0.05) + # Adjust binwidth
    vline_0(colour = "red", linetype = "dashed") +
    labs(
        title = "Posterior for Difference-in-Difference (Tau)",
        subtitle = "Change(Group B) - Change(Group A)",
        x = "Difference in Tau Change"
    )
print(plot_did_tau)
