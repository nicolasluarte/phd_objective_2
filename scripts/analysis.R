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
    theme_uncertainty +
    scale_x_discrete(labels = c("Ctrl", "Unc")) +
    scale_y_continuous(breaks = seq(-400,800,100), 
                       limits = c(-400,800), 
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
    theme_uncertainty +
    scale_x_discrete(labels = c("Ctrl", "Unc")) +
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
    scale_x_discrete(labels = c("Ctrl", "Unc")) +
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


