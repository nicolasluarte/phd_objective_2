# lib load ----
pacman::p_load(
    tidyverse,
    ggplot2,
    patchwork
)
setwd(this.path::here())

# Plot stuff ----
theme_uncertainty <- ggpubr::theme_pubr() +
    update_geom_defaults("point", list(size = 5, alpha = 0.5, shape = 21)) +
    update_geom_defaults("boxplot", list(width=0.5)) +
    theme(
        text = element_text(size = 24),
        axis.text=element_text(size=14),
        plot.margin = unit(c(0.5,0.5,0.5,0.5), "cm"),
        legend.position = "none"
    )

# Rewards (R) ----
rd <- read_csv("../datasets/lickometer_complete.csv")

## mdls ----

### number of licks ----
r_mdl1_d <- rd %>% 
    group_by(ID, fecha, exp_group, exp_phase) %>% 
    summarise(
        licks = n()
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    mutate(
        rel_date = as.numeric(fecha - min(fecha))
    )
r_mdl1_d

r_mdl1 <- lme4::glmer.nb(
    data = r_mdl1_d,
    licks ~ exp_group * exp_phase + (scale(rel_date)|ID),
    control = lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(r_mdl1)

r_mdl1_emm <- emmeans::emmeans(
    r_mdl1,
    pairwise ~ exp_group * exp_phase,
    type = "response"
)
r_mdl1_emm

### number of events ----
r_mdl2_d <- rd %>% 
    group_by(ID, fecha, exp_group, exp_phase, sensor) %>% 
    summarise(
        events = length(unique(evento)) - 1
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    mutate(
        rel_date = as.numeric(fecha - min(fecha))
    ) %>% 
    ungroup() %>% 
    group_by(ID, fecha, exp_group, exp_phase, rel_date) %>% 
    summarise(
        events = sum(events)
    )
r_mdl2_d


r_mdl2 <- lme4::glmer.nb(
    data = r_mdl2_d,
    events ~ exp_group * exp_phase + (scale(rel_date)|ID),
    control = lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(r_mdl2)

r_mdl2_emm <- emmeans::emmeans(
    r_mdl2,
    pairwise ~ exp_group * exp_phase,
    type = "response"
)
r_mdl2_emm

### number of rewards ----
r_mdl3_d <- rd %>% 
    group_by(ID, fecha, exp_group, exp_phase, sensor) %>% 
    summarise(
        rewards = length(unique(exito)) - 1
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    mutate(
        rel_date = as.numeric(fecha - min(fecha))
    ) %>% 
    ungroup() %>% 
    group_by(ID, fecha, exp_group, exp_phase, rel_date) %>% 
    summarise(
        rewards = sum(rewards)
    )
r_mdl3_d


r_mdl3 <- lme4::glmer.nb(
    data = r_mdl3_d,
    rewards ~ exp_group * exp_phase + (scale(rel_date)|ID),
    control = lme4::glmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(r_mdl3)

r_mdl3_emm <- emmeans::emmeans(
    r_mdl3,
    pairwise ~ exp_phase * exp_group,
    type = "response"
)
r_mdl3_emm

r_mdl4_d <- rd %>% 
    group_by(ID, fecha, exp_group, exp_phase, sensor) %>% 
    summarise(
        rewards = length(unique(exito)) - 1
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group, exp_phase) %>% 
    mutate(
        rel_date = as.numeric(fecha - min(fecha))
    ) %>% 
    ungroup() %>% 
    group_by(ID, fecha, exp_group, exp_phase, rel_date) %>% 
    summarise(
        rewards = sum(rewards)
    )
r_mdl4_d

### cluster analysis ----
r_mdl5_d <- rd %>% 
    select(ID, tiempo, tipo_recompensa, exp_phase, exp_group, fecha) %>% 
    group_by(ID, fecha) %>% 
    arrange(tiempo, .by_group = TRUE) %>% 
    mutate(
        ILI = tiempo-lag(tiempo, default = NA)
    ) %>% 
    ungroup() %>% 
    group_by(ID) %>% 
    mutate(
        rel_date = as.numeric(fecha-lag(fecha))
    ) %>% 
    filter(ILI>0)
r_mdl5_d


cluster_split <- r_mdl5_d %>% 
    group_by(ID, fecha) %>% 
    group_split()

# ILI dist
dist <- summary(r_mdl5_d$ILI)
cluster_thresholds <- seq(100, 1000, 100)

cluster_d <- 
    cluster_thresholds %>% 
    map_dfr(., function(outer){
        T <- outer
        map_dfr(cluster_split, function(X){
            ili_vec <- X %>% pull(ILI)
            ili_t <- as.numeric(ili_vec>T)
            inside_clust_idx <- which(ili_t==0)
            ili_clust <- cumsum(ili_t)[c(inside_clust_idx)]
            out <- tibble(
                ILI_idx = ili_clust
            ) %>% 
                mutate(ILI_idx=as.numeric(as.factor(ILI_idx))) %>% 
                group_by(ILI_idx) %>% 
                summarise(
                    clust_idx = head(ILI_idx,n=1),
                    clust_len = n()
                ) %>% 
                mutate(
                    ID = X$ID[1],
                    exp_phase = X$exp_phase[1],
                    exp_group = X$exp_group[1],
                    rel_date = X$rel_date[1],
                    T = T[1]
                )
            return(out)
        })
    })
cluster_d

cluster_mdls <-
    cluster_d %>% 
    group_by(T) %>% 
    group_split() %>% 
    map(., function(X){
        mdl <- lme4::glmer(
            data = X,
            clust_len ~ exp_group * exp_phase + (1|ID),
            family=Gamma(link="log"),
            control = lme4::glmerControl(
                optimizer = "bobyqa",
                optCtrl = list(maxfun = 2e5)
            )
        )
        return(mdl)
    })

cluster_n_mdls <-
    cluster_d %>% 
    group_by(T) %>% 
    group_split() %>% 
    map(., function(X){
        mdl <- lme4::glmer.nb(
            data = X %>% group_by(ID,exp_phase,exp_group) %>% 
                summarise(clust_n = n()),
            clust_n ~ exp_group * exp_phase + (1|ID),
            control = lme4::glmerControl(
                optimizer = "bobyqa",
                optCtrl = list(maxfun = 2e5)
            )
        )
        return(mdl)
    })

cluster_mdl_emm <- cluster_mdls %>% 
    imap_dfr(., function(mdl, idx){
        emmeans::emmeans(
            mdl,
            revpairwise ~ exp_group | exp_phase,
            type="response"
        )$contrasts %>% 
            broom.mixed::tidy(conf.int=TRUE) %>% 
            mutate(T = idx*100)
    })
cluster_mdl_emm

cluster_n_mdl_emm <- cluster_n_mdls %>% 
    imap_dfr(., function(mdl, idx){
        emmeans::emmeans(
            mdl,
            revpairwise ~ exp_group | exp_phase,
            type="response"
        )$contrasts %>% 
            broom.mixed::tidy(conf.int=TRUE) %>% 
            mutate(T = idx*100)
    })
cluster_n_mdl_emm

### licks per spout per phase ----
spout_d <- rd %>% 
    filter(exp_group == "experimental") %>% 
    group_by(ID, tipo_recompensa, exp_phase, fecha, sensor) %>% 
    summarise(
        licks = n()
    ) %>% 
    ungroup() %>% 
    group_by(ID) %>% 
    mutate(
        rel_date = as.numeric(fecha - head(fecha, n=1)),
        scaled_licks = scale(licks)) %>% 
    ungroup() %>% 
    mutate(
        reward_type = interaction(tipo_recompensa, exp_phase)
    )
spout_d

spout_mdl <- lme4::lmer(
    data = spout_d,
    licks ~ reward_type + rel_date + (1|ID),
    control = lme4::lmerControl(
        optimizer = "bobyqa",
        optCtrl = list(maxfun = 2e5)
    )
)
summary(spout_mdl)

spout_emm <- emmeans::emmeans(
    spout_mdl,
    revpairwise ~ reward_type + rel_date,
    type = "response"
)
spout_emm


### reinforcement learning ----

optimal_mdl_fit <- read_rds("../datasets/RL_model_fits.rds") %>% 
    group_by(ID, rel_date) %>% 
    slice(which.min(likelihood))

# check for exact 0s and 1s in alpha
sum(optimal_mdl_fit$opt_alpha == 0)
sum(optimal_mdl_fit$opt_alpha == 1)
# the same for tau after transformation to 0 and 1
sum(optimal_mdl_fit$opt_tau/5 == 0)
sum(optimal_mdl_fit$opt_tau/5 == 1)


rl_mdl_data <- optimal_mdl_fit %>% 
    ungroup() %>% 
    mutate(
        learning_rate = (opt_alpha * (n() - 1) + 0.5) / n(),
        temperature = ((opt_tau/5)* (n() - 1) + 0.5) / n()
    ) %>% 
    group_by(ID) %>% 
    arrange(exp_phase, rel_date, .by_group = TRUE)


# beta_model_transformed <- glmmTMB(
#     transformed_dv ~ predictor1 + (random | group), # Use the transformed DV
#     data = your_data,
#     family = beta_family(link = "logit")
# )

# used beta to inform the model of hard boundaries impose in the
# optimization procedure
temperature_mdl <- glmmTMB::glmmTMB(
    data = rl_mdl_data,
    temperature ~ exp_group * exp_phase + rel_date  + (1|ID),
    family = glmmTMB::beta_family(link="logit")
)
summary(temperature_mdl)

temperature_mdl_emm <- emmeans::emmeans(
    temperature_mdl,
    revpairwise ~ exp_group | exp_phase + rel_date,
    type = "response"
)
temperature_mdl_emm

# same idea for the learning rate
lrate_mdl <- glmmTMB::glmmTMB(
    data = rl_mdl_data,
    learning_rate ~ exp_group * exp_phase + rel_date + (1|ID),
    family = glmmTMB::beta_family(link="logit")
)
summary(lrate_mdl)

lrate_mdl_emm <- emmeans::emmeans(
    lrate_mdl,
    revpairwise ~ exp_group | exp_phase + rel_date,
    type = "response"
)
lrate_mdl_emm

# correlation between parameters
corr_rl_mdl <- glmmTMB::glmmTMB(
    data = rl_mdl_data,
    temperature ~ exp_group * exp_phase * learning_rate + rel_date + (1|ID),
    family = glmmTMB::beta_family(link="logit")
)
summary(corr_rl_mdl)

corr_rl_mdl_emm <- emmeans::emmeans(
    corr_rl_mdl,
    revpairwise ~ exp_group | learning_rate * exp_phase + rel_date,
    type = "response"
)
corr_rl_mdl_emm

corr_rl_mdl_emtrends <- emmeans::emtrends(
    corr_rl_mdl,
    revpairwise ~ exp_group | learning_rate * exp_phase + rel_date,
    type = "response",
    var = "learning_rate"
)
corr_rl_mdl_emtrends



## p::rl learning rate ----
lrp1 <- lrate_mdl_emm$contrasts %>% 
    broom.mixed::tidy(conf.int=TRUE) %>% 
    ggplot(aes(
        exp_phase, odds.ratio-1
    )) +
    geom_pointrange(aes(ymin=conf.low-1, ymax=conf.high-1,
                        color=exp_phase), size=1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    annotate("text", label="*", x=seq(1, 2), y=0.95, size=c(0,12)) +
    theme_uncertainty +
    scale_x_discrete(labels = c("Bsl.", "Exp.")) +
    scale_y_continuous(breaks = seq(-0.25,1.25,0.25), 
                       limits = c(-0.25, 1.25), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\alpha_{\frac{HU-LU}{LU}}$)")) +
    xlab("") +
    scale_color_manual(values = c("black", "orange")) 
lrp1

## p::rl temperature ----
lrp2 <- temperature_mdl_emm$contrasts %>% 
    broom.mixed::tidy(conf.int=TRUE) %>% 
    ggplot(aes(
        exp_phase, odds.ratio-1
    )) +
    geom_pointrange(aes(ymin=conf.low-1, ymax=conf.high-1,
                        color=exp_phase), size=1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    annotate("text", label="*", x=seq(1, 2), y=1.90, size=c(0,12)) +
    theme_uncertainty +
    scale_x_discrete(labels = c("Bsl.", "Exp.")) +
    scale_y_continuous(breaks = seq(-0.5,2.25,0.25), 
                       limits = c(-0.5, 2.25), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\tau_{\frac{HU-LU}{LU}}$)")) +
    xlab("") +
    scale_color_manual(values = c("black", "orange")) 
lrp2

## p::rl learning_rate/temperature ----
lrp3 <- corr_rl_mdl_emm$contrasts %>% 
    broom.mixed::tidy(conf.int=TRUE) %>% 
    ggplot(aes(
        exp_phase, odds.ratio-1
    )) +
    geom_pointrange(aes(ymin=conf.low-1, ymax=conf.high-1,
                        color=exp_phase), size=1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    annotate("text", label="*", x=seq(1, 2), y=2.20, size=c(0,12)) +
    theme_uncertainty +
    scale_x_discrete(labels = c("Bsl.", "Exp.")) +
    scale_y_continuous(breaks = seq(-0.5,2.5,0.5), 
                       limits = c(-0.5, 2.5), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($(\tau | \alpha)_{\frac{HU-LU}{LU}}$)")) +
    xlab("") +
    scale_color_manual(values = c("black", "orange")) 
lrp3

lrp4 <- corr_rl_mdl_emtrends$contrasts %>% 
    broom.mixed::tidy(conf.int=TRUE) %>% 
    ggplot(aes(
        exp_phase, estimate
    )) +
    geom_pointrange(aes(ymin=conf.low-1, ymax=conf.high-1,
                        color=exp_phase), size=1.25) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    theme_uncertainty +
    scale_x_discrete(labels = c("Bsl.", "Exp.")) +
    scale_y_continuous(breaks = seq(-4,3,1), 
                       limits = c(-4, 3), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($(\tau | \alpha)_{HU_{slope}-LU_{slope}}$)")) +
    xlab("") +
    scale_color_manual(values = c("black", "orange")) 
lrp4



## p::licks per spout per phase ----

pred_grid <- spout_d %>% 
    distinct(ID, reward_type) %>% 
    mutate(rel_date = 16,
           preds = predict(spout_mdl, newdata = pred_grid))
pred_grid

sp1 <- spout_emm$emmeans %>% 
    broom.mixed::tidy(conf.int = TRUE) %>% 
    ggplot(aes(
        reward_type, estimate
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high), size=1.25,
                    color = c("gray", "orange", "orange")) +
    geom_point(data = pred_grid, aes(reward_type, preds, fill=reward_type)) +
    ggsignif::geom_signif(
        comparisons = list(c(1,2), c(3,2)),
        annotations = c("", ""),
        y_position = c(900, 950)
    ) +
    theme_uncertainty +
    scale_x_discrete(labels = c(
        latex2exp::TeX(r"($P=1_{basal}$)"),
        latex2exp::TeX(r"($P=1_{experimental}$)"),
        latex2exp::TeX(r"($P=0.5_{experimental}$)")
    )) +
    scale_y_continuous(breaks = seq(200, 1000, 200), 
                       limits = c(200, 1000), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\beta_{licks}$)")) +
    xlab("") +
    scale_fill_manual(values = c("gray", "orange", "orange")) 
sp1
    



## p::licks per group/phase ----
rp1 <- r_mdl1_d %>% 
    ungroup() %>% 
    group_by(ID, exp_group, exp_phase) %>% 
    summarise(
        licks = mean(licks)
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    summarise(
        delta_licks = licks[exp_phase=="experimental"]-licks[exp_phase=="basal"]
    ) %>% 
    ggplot(aes(
        exp_group, delta_licks
    )) +
    geom_boxplot(outlier.shape = NA, width=0.5, aes(color=exp_group)) +
    geom_point(aes(fill=exp_group), color="black") +
    geom_hline(yintercept = 0, linetype = 'dashed') +
    annotate("text", label="*", x=seq(1, 2), y=800, size=c(0,12)) +
    theme_uncertainty +
    scale_x_discrete(labels = c("LU", "HU")) +
    scale_y_continuous(breaks = seq(-400,1000,200), 
                       limits = c(-400,1000), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\Delta Licks_{Bl - Exp}$)")) +
    xlab("") +
    scale_fill_manual(values = c("black", "orange")) +
    scale_color_manual(values = c("black", "orange")) 
rp1

## p::events per group/phase ----
rp2 <- r_mdl2_d %>% 
    ungroup() %>% 
    group_by(ID, exp_group, exp_phase) %>% 
    summarise(
        events = mean(events)
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    summarise(
        delta_events = events[exp_phase=="experimental"]-events[exp_phase=="basal"]
    ) %>% 
    ggplot(aes(
        exp_group, delta_events
    )) +
    geom_boxplot(outlier.shape = NA, width=0.5, aes(color=exp_group)) +
    geom_point(aes(fill=exp_group), color="black") +
    geom_hline(yintercept = 0, linetype = 'dashed') +
    annotate("text", label="*", x=seq(1, 2), y=30, size=c(0,12)) +
    theme_uncertainty +
    scale_x_discrete(labels = c("LU", "HU")) +
    scale_y_continuous(breaks = seq(-20,40,10), 
                       limits = c(-20,40), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\Delta Events_{Bl - Exp}$)")) +
    xlab("") +
    scale_fill_manual(values = c("black", "orange")) +
    scale_color_manual(values = c("black", "orange")) 
rp2

## p::rewards per group/phase ----
rp3 <- r_mdl3_d %>% 
    ungroup() %>% 
    group_by(ID, exp_group, exp_phase) %>% 
    summarise(
        rewards = mean(rewards)
    ) %>% 
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    summarise(
        delta_rewards = rewards[exp_phase=="experimental"]-rewards[exp_phase=="basal"]
    ) %>% 
    ggplot(aes(
        exp_group, delta_rewards
    )) +
    geom_boxplot(outlier.shape = NA, width=0.5, aes(color=exp_group)) +
    geom_point(aes(fill=exp_group), color="black") +
    geom_hline(yintercept = 0, linetype = 'dashed') +
    theme_uncertainty +
    scale_x_discrete(labels = c("LU", "HU")) +
    scale_y_continuous(breaks = seq(-20,40,10), 
                       limits = c(-20,40), 
                       expand = c(0,0)) +
    ylab(latex2exp::TeX(r"($\Delta Rewards_{Bl - Exp}$)")) +
    xlab("") +
    scale_fill_manual(values = c("black", "orange")) +
    scale_color_manual(values = c("black", "orange")) 
rp3

## p::reward training session ----
rp4 <- r_mdl4_d %>% 
    ungroup() %>% 
    filter(exp_phase == "basal", rel_date<=11) %>% # trained by at least 11 sessions
    ungroup() %>% 
    group_by(ID, exp_group) %>% 
    ggplot(aes(
        rel_date, rewards
    )) +
    geom_point(aes(fill=exp_group), color="black") +
    stat_summary(
        aes(group=exp_group, fill=exp_group),
        fun.data = "mean_se",
        geom = "ribbon",
        alpha=0.5
    ) +
    theme_uncertainty +
    scale_x_continuous(breaks = seq(0, 11, 1)) +
    scale_y_continuous(breaks = seq(0,100,10), 
                       limits = c(0,100), 
                       expand = c(0,0)) +
    ylab("# Rewards") +
    xlab("Training sessions") +
    scale_fill_manual(values = c("black", "orange")) +
    scale_color_manual(values = c("black", "orange")) 
rp4

## p::cluster size ----
star_y_cp1 <- cluster_mdl_emm %>% filter(exp_phase=="experimental") %>% pull(conf.high)
pval_cp1 <- cluster_mdl_emm %>% filter(exp_phase=="experimental") %>%
    mutate(p.value=if_else(p.value<0.05,12,0)) %>% pull(p.value)
cp1 <- cluster_mdl_emm %>% 
    filter(exp_phase=="experimental") %>% 
    ggplot(aes(
        T, ratio
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high), color="orange") +
    geom_hline(yintercept = 1, linetype="dashed") +
    annotate("text", label="*", x=seq(100,1000,100), y=star_y_cp1, size=pval_cp1) +
    theme_uncertainty +
    scale_y_continuous(breaks = seq(0,3.5,0.5), 
                       limits = c(0,3.5), 
                       expand = c(0,0)) +
    scale_x_continuous(breaks=seq(100, 1000, 100)) +
    ylab(latex2exp::TeX(r"($Clst. size \ \beta_{HU/LU}$)")) +
    xlab("Cluster threshold (ms.)") 
cp1

## p::cluster number ----
star_y_cp2 <- cluster_n_mdl_emm %>% filter(exp_phase=="experimental") %>% pull(conf.high)
pval_cp2 <- cluster_n_mdl_emm %>% filter(exp_phase=="experimental") %>%
    mutate(p.value=if_else(p.value<0.05,12,0)) %>% pull(p.value)
cp2 <- cluster_n_mdl_emm %>% 
    filter(exp_phase=="experimental") %>% 
    ggplot(aes(
        T, ratio
    )) +
    geom_pointrange(aes(ymin=conf.low, ymax=conf.high), color="orange") +
    geom_hline(yintercept = 1, linetype="dashed") +
    annotate("text", label="*", x=seq(100,1000,100), y=star_y_cp2, size=pval_cp2) +
    theme_uncertainty +
    scale_y_continuous(breaks = seq(0,4,0.5), 
                       limits = c(0,4), 
                       expand = c(0,0)) +
    scale_x_continuous(breaks=seq(100, 1000, 100)) +
    ylab(latex2exp::TeX(r"($Clst. number \ \beta_{HU/LU}$)")) +
    xlab("Cluster threshold (ms.)") 
cp2

# figures ----

r1 <- (rp1 + rp2 + rp3)
r2 <- (cp1 + cp2 + sp1)
r3 <- (lrp1 + lrp2)
r4 <-  (lrp3 | lrp4)

r1 / r2 / (r3 + r4)


