pacman::p_load(
   cmdstanr,
   tidyverse,
   posterior
)

setwd(this.path::here())
# see RL_model script if you need to modify data
stan_data <- read_rds("../datasets/stan_data.rds")

stan_data_list <- list(
    N = nrow(stan_data),
    P = length(unique(stan_data$id)),
    n_choices = 2,
    n_groups = length(unique(stan_data$treatment)),
    group_id = stan_data %>% ungroup %>% distinct(id,treatment) %>% pull(treatment),  # Maps subject p -> group g
    subj_id = stan_data$p,     # Maps trial n -> subject p
    phase_id = stan_data$phase,   # Maps trial n -> phase ph
    day_id = stan_data$day+1, # <<< NEW: Maps trial n -> unique day number for subject
    actions = stan_data$action,     # Action for trial n
    rewards = stan_data$reward      # Reward for trial n
)

stan_file <- "rl_hbm.stan" # Use the new Stan file name
mod <- cmdstan_model(stan_file, compile = TRUE)

# just to check if model CAN run
fit_test <- mod$sample(
    data = stan_data_list, seed = 123, chains = 1, parallel_chains = 1, # Use only 1 chain first
    iter_warmup = 50, iter_sampling = 50, refresh = 5
)

bayesian_fit <- mod$sample(
    data = stan_data_list,
    seed = 1234567,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000, # Consider increasing warmup
    iter_sampling = 2000, # Consider increasing sampling
    refresh = 200,
    adapt_delta = 0.95, # Higher value often needed
    max_treedepth = 12 # Might need increasing
)

