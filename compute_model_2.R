# In 1. the state is (current_demand, current_storage) and the action is (storage_change, production_level)
# production_level is bounded in [0,1]
# storage_change is bounded in [-1,+1] and
#   when positive, represents the percentage of production stored
#   when negative, represents the percentage of storage unstored
#   hence this is a relative percentage
# -> this is equivalent to deciding whether to store or unstore 
# and defining the two percentages with the condition that both of them
# can't be strictly positive simultaneously.
#
# **important note**: demand is independent from the storage decisions :)

### PARAMETERS ====

# set of known constants in equations (1), (2), (3) and constraints on \tau
problem_constants <- list(
  'a' = 300, # slope of (1)
  'b' = 100, # intercept of (1)
  'c' = 0.9, # production efficiency of our units (2)
  'd' = 10,  # unit cost of input energy (3)
  'e' = 100, # 100,  # unit cost / penalty due to lack of supply (3)
  'f' = 20,   # unit reward per production output
  'tau_min' = 0.2, # minimal functioning level
  'tau_max' = 0.9  # maximal functioning level
)
problem_constants$P_min <- with(problem_constants, c*(a*tau_min + b))
problem_constants$P_max <- with(problem_constants, c*(a*tau_max + b)) 
pars <- problem_constants

### PROBLEM PARAMETERS ====

# nb of data points available for training
NN <- 2000

# demand range
MAX_DEMAND <- 400
MIN_DEMAND <- 100

# storage range
MAX_STORAGE <- 400
MIN_STORAGE <- 0

### SIMULATE DATASETS ====

# demand ~ UNIFORM
set.seed(1785)
past_demand_sequence <- runif(n = NN, min = MIN_DEMAND, max = MAX_DEMAND)
demand_sequence <- pmax( pmin( (1 - 0.8) * 238.5 + 0.8 * past_demand_sequence + 10.79 * sqrt(5) * rnorm(n = NN, mean = 0, sd = 1), MAX_DEMAND), MIN_DEMAND)
next_demand_sequence <- pmax( pmin( (1 - 0.8) * 238.5 + 0.8 * demand_sequence + 10.79 * sqrt(5) * rnorm(n = NN, mean = 0, sd = 1), MAX_DEMAND), MIN_DEMAND)

# storage ~ UNIFORM
set.seed(3964)
initial_storage_sequence <- runif(n = NN, min = MIN_STORAGE, max = MAX_STORAGE) 

# initialize state
initial_state <- cbind(
  'past_demand' = past_demand_sequence,
  'demand' = demand_sequence,
  'storage' = initial_storage_sequence
)

# initialize decisions made
decision_sequence <- cbind(
  'storage_change' = runif(n = NN, min = -1, max = +1), 
  'production_level' = runif(n = NN, min = pars$tau_min, max = pars$tau_max)
)

# compute intermediaries
to_compute_reward <- data.frame(
  'past_demand' = initial_state[,'past_demand'],
  'demand' = initial_state[,'demand'],
  'It' = pars$a * decision_sequence[,'production_level'] + pars$b
)
to_compute_reward$Pt <- pars$c * to_compute_reward$It
cond <- (decision_sequence[,'storage_change'] > 0)
to_compute_reward$Storedt <- 0
to_compute_reward$Unstoredt <- 0
to_compute_reward$Storedt[cond] <- (decision_sequence[,'storage_change'] * to_compute_reward$Pt)[cond]
to_compute_reward$Unstoredt[cond == FALSE] <- (- decision_sequence[,'storage_change'] * initial_state[,'storage'])[cond == FALSE]
to_compute_reward$Supplyt <- to_compute_reward$Pt - to_compute_reward$Storedt + to_compute_reward$Unstoredt

# compute reward
to_compute_reward$REWARD <- pars$d * to_compute_reward$It + 
  pars$e * pmax(to_compute_reward$demand - to_compute_reward$Supplyt, 0) - 
  pars$f * pmin(to_compute_reward$demand, to_compute_reward$Supplyt)

# compute next state
next_state <- cbind(
  'demand' = next_demand_sequence,
  'storage' = pmin(initial_storage_sequence + to_compute_reward$Storedt - to_compute_reward$Unstoredt, MAX_STORAGE)
)

### INITIALIZE NETWORK ====

# load libraries
library(keras)
library(tensorflow)

# input / output network dim
input_size <- 2 + 2 # 2 for state, 2 for action
output_size <- 1

# from state of layer 2 + action to final reward || architecture + network instanciation + compilation
fl <- 30
network2 <- keras_model_sequential() %>%
  layer_dense(units = fl, kernel_initializer = "uniform", input_shape = input_size, activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = fl, kernel_initializer = "uniform", activation = "relu") %>%
  layer_dense(units = output_size, kernel_initializer = "uniform", activation = "linear")
network2 %>% compile(optimizer = optimizer_rmsprop(lr = 1e-4), loss = "mse")
network2$optimizer$get_config()
summary(network2)

# prepare callbacks for training
fit_callbacks <- list(
  callback_early_stopping(monitor = "val_loss", min_delta = 1e-5, patience = 25, verbose = 0, mode = "auto"),
  callback_model_checkpoint(filepath = ".mdl_wts.hdf5", monitor = "val_loss", verbose = 0, save_best_only = TRUE, mode = "min"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", min_delta = 1e-5, factor = 0.99, patience = 10, mode = "auto")
)

### LEARN Q-values ====

# 
gamma <- 0.75
## sum_{k = 0 ... t} gamma^k \geq p / (1 - gamma)
## l.h.s = (1 - gamma^t) / (1 - gamma)
## hence: (1 - gamma^t) \geq p
##        1-p \geq gamma^t
##        log(1-p)/\log(gamma) \geq t 
horiz_levels <- sapply(X = seq(0, 1, by = 0.1), FUN = function(p){
  (log(1 - p) / log(gamma))
})
plot.ts(cumsum(gamma^(0:100)), ylim = c(0, 1/(1 - gamma)))
abline(v = horiz_levels, lty = 3)

# learn 
#
for(iteration in 1:10){
  
  cat(paste0(
    '\n\n ********************************************',
    '\n\n Iteration \t', iteration,
    '\n\n ******************************************** \n\n'
  ))
  
  #
  next_to_compute_reward <- data.frame(
    'past_demand' = next_state[,'past_demand'],
    'storage' = next_state[,'storage'],
    'storage_change' = NA,
    'production_level' = NA
  )
  
  #
  sqrt_nb_knots <- 25
  storage_change_cross_prod <- expand.grid(
    'storage_change' = seq(-1, +1, length.out = sqrt_nb_knots), 
    'production_level' = seq(pars$tau_min, pars$tau_max, length.out = sqrt_nb_knots)
  )
  ## dim(storage_change_cross_prod)
  
  #
  finding_best_action <- sapply(X = 1:nrow(storage_change_cross_prod), FUN = function(j){
    next_to_compute_reward[,3] <- storage_change_cross_prod$storage_change[j]
    next_to_compute_reward[,4] <- storage_change_cross_prod$production_level[j]
    predict(network2, x = as.matrix(next_to_compute_reward))
  })
  
  #
  qval_next_estimate <- apply(finding_best_action, 1, min)
  range(qval_next_estimate, na.rm = T)
  
  #
  new_reward <- (to_compute_reward$REWARD + gamma * qval_next_estimate) / 1000
  
  # next_to_compute_reward$It = pars$a * next_to_compute_reward$production_level + pars$b
  # next_to_compute_reward$Pt <- pars$c * next_to_compute_reward$It
  # cond <- (next_to_compute_reward$storage_change > 0)
  # next_to_compute_reward$Storedt <- 0
  # next_to_compute_reward$Unstoredt <- 0
  # next_to_compute_reward$Storedt[cond] <- (next_to_compute_reward$storage_change * next_to_compute_reward$Pt)[cond]
  # next_to_compute_reward$Unstoredt[cond == FALSE] <- (- next_to_compute_reward$storage_change * next_to_compute_reward$storage)[cond == FALSE]
  
  # fit model again :)
  network2 %>% fit(x = cbind(initial_state, decision_sequence)[-NN,], 
                   y = new_reward[-NN], 
                   epochs = 500, 
                   validation_split = 0.5, 
                   callbacks = fit_callbacks)
  
}

### DERIVE (FINAL) POLICY ====

# grid of actions
storage_change_cross_prod <- expand.grid(
  'storage_change' = seq(-1, +1, length.out = 50), 
  'production_level' = seq(pars$tau_min, pars$tau_max, length.out = 50)
)

# grid of states
bingaloo <- expand.grid('past_demand' = seq(min(initial_state[,'past_demand']), max(initial_state[,'past_demand']), length.out = 50),
                        'storage' = seq(min(initial_state[,'storage']), max(initial_state[,'storage']), length.out = 50))

# for each state we try the potential actions and observe Q-values
finding_best_action <- sapply(X = 1:nrow(storage_change_cross_prod), FUN = function(j){
  bingaloo[,3] <- storage_change_cross_prod$storage_change[j]
  bingaloo[,4] <- storage_change_cross_prod$production_level[j]
  predict(network2, x = as.matrix(bingaloo))
})

# find best action and retrieve informations
best_action_considered_not_last_one <- apply(finding_best_action, 1, which.min)
bingaloo$result_index <- best_action_considered_not_last_one
bingaloo$storage_change <- storage_change_cross_prod$storage_change[bingaloo$result_index]
bingaloo$production_level <- storage_change_cross_prod$production_level[bingaloo$result_index]

### SAVE INITIAL DATA AND POLICY ====

saveRDS(object = bingaloo, file = 'model_2_policy.RDS')
initial_state_df <- cbind.data.frame(initial_state, next_state)
colnames(initial_state_df) <- c('state_past_demand', 'state_demand', 'state_storage', 'next_state_demand', 'next_state_storage')
saveRDS(object = initial_state_df, file = 'model_2_data_generated.RDS')
