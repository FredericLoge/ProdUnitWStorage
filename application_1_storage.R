# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#
# OBJECTIVES
#
# build algorithm which from true immediate future demand 
# proposes the best storage option for long term 
# (by reinforcement for example)
# then, plug the tweaking linearly the best estimator
#
# finally, as comparison, optimize jointly the storage 
# and production level
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# 1. evaluate best (storage, production_level) decision assuming next day demand
# is available to us
# 2. then treat the demand forecast input to do tweaks
#
# In 1. the state is (current_demand, current_storage) and the action is (storage_change, production_level)
# production_level is bounded in [0,1]
# storage_change is bounded in [-1,+1] and
#   when positive, represents the percentage of production stored
#   when negative, represents the percentage of storage unstored
#   hence this is a relative percentage
# -> this is equivalent to deciding whether to store or unstore 
# and defining the two percentages with the condition that both of them
# can't be strictly positive simultaneously.

# **important note**: demand is independent from the storage decisions :)

NN <- 20000

set.seed(1785)
MAX_DEMAND <- 400
MIN_DEMAND <- 100
demand_sequence <- runif(n = NN, min = MIN_DEMAND, max = MAX_DEMAND)
## demand_sequence <- sim_ar_1(n = NN, sigma2e = 5, theta = 0.8)
MAX_STORAGE <- 400 # max(demand_sequence)
MIN_STORAGE <- 0
initial_storage_sequence <- runif(n = NN, min = MIN_STORAGE, max = MAX_STORAGE) 

initial_state <- cbind(
  'demand' = demand_sequence,
  'storage' = initial_storage_sequence
)

decision_sequence <- cbind(
  'storage_change' = runif(n = NN, min = -1, max = +1), 
  'production_level' = runif(n = NN, min = pars$tau_min, max = pars$tau_max)
)

to_compute_reward <- data.frame(
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

to_compute_reward$REWARD <- pars$d * to_compute_reward$It + 
  pars$e * pmax(to_compute_reward$demand - to_compute_reward$Supplyt, 0) - 
  pars$f * pmin(to_compute_reward$demand, to_compute_reward$Supplyt)

next_state <- cbind(
  'demand' = pmax( pmin( (1 - 0.8) * 238.5 + 0.8 * demand_sequence + 10.79 * sqrt(5) * rnorm(n = NN, mean = 0, sd = 1), MAX_DEMAND), MIN_DEMAND),
  'storage' = pmin(initial_storage_sequence + to_compute_reward$Storedt - to_compute_reward$Unstoredt, MAX_STORAGE)
)

# (state - action - reward - next state)

library(keras)
library(tensorflow)

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

# initial fit, gamma = 0
fit_callbacks <- list(
  callback_early_stopping(monitor = "val_loss", min_delta = 1e-5, patience = 25, verbose = 0, mode = "auto"),
  callback_model_checkpoint(filepath = ".mdl_wts.hdf5", monitor = "val_loss", verbose = 0, save_best_only = TRUE, mode = "min"),
  callback_reduce_lr_on_plateau(monitor = "val_loss", min_delta = 1e-5, factor = 0.99, patience = 10, mode = "auto")
)
### network2 %>% fit(x = cbind(initial_state, decision_sequence), y = to_compute_reward$REWARD / 1000, epochs = 1000, validation_split = 0.5, callbacks = fit_callbacks)

# 
gamma <- 0.75
ceiling(log(0.9 / (1 - gamma)) / log(1/gamma))
plot.ts(cumsum(gamma^(0:100)))
abline(v = 56)

#
sqrt_nb_knots <- 10

#
for(iteration in 12:100){
  
  cat(paste0(
    '\n\n ********************************************',
    '\n\n Iteration \t', iteration,
    '\n\n ******************************************** \n\n'
  ))
  
  #
  next_to_compute_reward <- data.frame(
    'demand' = next_state[,'demand'],
    'storage' = next_state[,'storage'],
    'storage_change' = NA,
    'production_level' = NA
  )
  
  #
  storage_change_cross_prod <- expand.grid('storage_change' = seq(-1, +1, length.out = sqrt_nb_knots), 'production_level' = seq(pars$tau_min, pars$tau_max, length.out = sqrt_nb_knots))
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
  network2 %>% fit(x = cbind(initial_state, decision_sequence)[-NN,], y = new_reward[-NN], epochs = 1000, validation_split = 0.5, callbacks = fit_callbacks)
  
}

#
storage_change_cross_prod <- expand.grid(
  'storage_change' = seq(-1, +1, length.out = 50), 
  'production_level' = seq(pars$tau_min, pars$tau_max, length.out = 50)
)
bingaloo <- expand.grid('demand' = seq(min(initial_state[,'demand']), max(initial_state[,'demand']), length.out = 50),
                        'storage' = seq(min(initial_state[,'storage']), max(initial_state[,'storage']), length.out = 50))
finding_best_action <- sapply(X = 1:nrow(storage_change_cross_prod), FUN = function(j){
  bingaloo[,3] <- storage_change_cross_prod$storage_change[j]
  bingaloo[,4] <- storage_change_cross_prod$production_level[j]
  predict(network2, x = as.matrix(bingaloo))
})
best_action_considered_not_last_one <- apply(finding_best_action, 1, which.min)
bingaloo$result_index <- best_action_considered_not_last_one
bingaloo$storage_change <- storage_change_cross_prod$storage_change[bingaloo$result_index]
bingaloo$production_level <- storage_change_cross_prod$production_level[bingaloo$result_index]
str(bingaloo)
library(lattice)
wireframe(production_level ~ demand * storage, data = bingaloo,
          scales = list(arrows = FALSE),
          drape = TRUE, colorkey = TRUE,
          screen = list(z = 30, x = -60))
wireframe(storage_change ~ demand * storage, data = bingaloo,
          scales = list(arrows = FALSE),
          drape = TRUE, colorkey = TRUE,
          screen = list(z = 30, x = -60))
library(ggplot2)
graph_opt_prod <- ggplot(data = bingaloo) +
  aes(x = demand, y = storage, fill = production_level) +
  geom_tile() +
  scale_fill_viridis_c()
graph_opt_storage <- ggplot(data = bingaloo) +
  aes(x = demand, y = storage, fill = storage_change) +
  geom_tile() +
  scale_fill_viridis_c()

initial_state_df <- data.frame(initial_state)
some_data_points <- ggplot(data = initial_state_df) +
  aes(x = demand, y = storage) +
  geom_hex() +
  scale_fill_viridis_c()

library(gridExtra)
gridExtra::grid.arrange(graph_opt_prod, graph_opt_storage, some_data_points, layout_matrix = matrix(c(1, NA, 2, 3), byrow = TRUE, ncol = 2))

# now:
#
# evaluate on simulated data the performance of the derived policy ... hopefully better than the classic linear model
# without storage facility ...
#

set.seed(2742)

NN_TEST <- 100
MAX_DEMAND
MIN_DEMAND
demand_sequence_TEST <- sim_ar_1(n = NN_TEST, sigma2e = 0.5, theta = 0.8)
MAX_STORAGE <- 400 # max(demand_sequence)
MIN_STORAGE <- 0
initial_storage_TEST <- c(runif(n = 1, min = MIN_STORAGE, max = MAX_STORAGE), rep(NA, NN_TEST - 1))

test_df <- data.frame(
  'demand' = demand_sequence_TEST,
  'storage' = initial_storage_TEST,
  'decision_storage_change' = NA,
  'decision_production_level' = NA,
  "It" = NA, 
  "Pt" = NA,
  "Storedt" = 0,
  "Unstoredt" = 0,
  "Supplyt" = NA,
  "REWARD" = NA
)

## construct my_fooo() which takes min from the  
# diff(sort(unique(bingaloo$demand)))
# diff(sort(unique(bingaloo$storage)))
my_fooo <- function(demand, storage){
  valeur_plus_proche_demand <- bingaloo$demand[which.min(abs(bingaloo$demand - demand))]
  valeur_plus_proche_storage <- bingaloo$storage[which.min(abs(bingaloo$storage - storage))]
  indexx <- which( (abs(bingaloo$demand - valeur_plus_proche_demand) < 1e-10) & (abs(bingaloo$storage - valeur_plus_proche_storage) < 1e-10) )
  c("storage_change" = bingaloo$storage_change[indexx],
    "production_level" = bingaloo$production_level[indexx])
}
my_fooo(demand = 200, storage = 0)

for(i in 1:NN_TEST){
  
  # compute decision and store it in dataframe
  my_decision <- my_fooo(demand = test_df$demand[i], storage = test_df$storage[i])
  test_df$decision_storage_change[i] <- my_decision['storage_change']
  test_df$decision_production_level[i] <- my_decision['production_level']
  
  # compute intermediaries
  test_df$It[i] = pars$a * test_df$decision_production_level[i] + pars$b
  test_df$Pt[i] <- pars$c * test_df$It[i]
  if(test_df$decision_storage_change[i] > 0){
    test_df$Storedt[i] <- test_df$decision_storage_change[i] * test_df$Pt[i]
  }else{
    test_df$Unstoredt[i] <- (- test_df$decision_storage_change[i]) * test_df$storage[i]
  }
  test_df$Supplyt[i] <- test_df$Pt[i] - test_df$Storedt[i] + test_df$Unstoredt[i]
  
  # compute reward
  test_df$REWARD[i] <- pars$d * test_df$It[i] + 
    pars$e * pmax(test_df$demand[i] - test_df$Supplyt[i], 0) - 
    pars$f * pmin(test_df$demand[i], test_df$Supplyt[i])
  
  # compute next storage value
  if(i < NN_TEST){
    test_df$storage[i+1] <- test_df$storage[i] + test_df$Storedt[i] - test_df$Unstoredt[i]
  }
  
}

mean(test_df$REWARD)
summary(test_df$REWARD)

#
#
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# Consider the following control problem.
#
# \tau, representing the level of use of the machines in the prod unit is our only control variable.
# \tau is bounded in [\tau_{min} ; \tau_{max}].
#
# Q (the elec quantity needed) and P (the production in output) are given by : 
#   Q <- a*\tau + b     (1)
#   P <- c*Q            (2)
# with (a,b,c) known quantities. (a,b) represent the conversion between the use level and the electric
# needs, (c) represents the production efficiency of our units.
#
# We define the cost function :
#   L(\tau, D) = d*(P-D)_{+} + e*(P-D)_{-}, (3)
# where (x)_{+} = max(0,x), (x)_{-} = max(0,-x) and (d,e) known quantities. d is the unit cost of vented
# molecules (P > D), e is the unit cost / penalty due to lack of supply. 
#
# To complete this control problem, we add the following assumptions :
# - at each decision epoch, we decide of the value of \tau for the next time step
# - the optimal decision would be based on the actual demand on this time step, but since we don't have
#   access to it, we will forecast this demand based on prior demands.
#
# Note that P <- c*(a*\tau + b), combining (1) and (2). As such, the range of demand which the
# production may handle is [c*(a*\tau_{min} + b) ; c*(a*\tau_{max} + b)].

# Here is the content of this R file :
# - problem constants from equations above
# - number of time units
# - simulate demand signal under AR(p)
# - solve two optim problems -> "best" forecasters
# - graphic representation of the loss functions
# - evaluation on test set

# needs installation prior :p
library(taskDrivenRandomForest)
source('data_sim.R')
source('custom_metrics.R')
source('task_driven_lm.R')

### PROBLEM CONSTANTS FROM AFOREMENTIONED EQUATIONS -------------------------------------------------

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
pars <- problem_constants

# add to set problem_constants the bounds of production capacity
problem_constants$P_min <- with(problem_constants, c*(a*tau_min + b))
problem_constants$P_max <- with(problem_constants, c*(a*tau_max + b)) 

s_range <- sqrt( 2 * log(2 * 5000 / 0.01) / (1 - 0.8^2))
(problem_constants$P_max - problem_constants$P_min) / (2 * s_range)
problem_constants$P_min + (problem_constants$P_max - problem_constants$P_min) / 2

with(problem_constants, c(b * (1-0.8), 0.8))

library(tidyverse)

#
paste0(floor(runif(n = 10, min = 1e3, max = 1e4)), collapse = ', ')

list_of_seeds <- c(7611, 9575, 3963, 9458, 8035, 2231, 5550, 2865, 2880, 4524)
resss <- list()

for(iter in 1:length(list_of_seeds)){
  
  cat('\n\n iter: ', iter)
  set.seed(list_of_seeds[iter])
  
  ### SIMULATE DATA ---------------------------------------------------------------------------------
  
  # number of time units
  n <- 5000
  SIGNAL_TYPE = 2
  NOISE_TYPE = 2
  
  # simulate signal 
  signal <- sim_signal(n = n, signal_type = SIGNAL_TYPE)
  
  # simulate noise 
  noise <- sim_noise(n = n, noise_type = NOISE_TYPE)
  
  # add signal and noise, weigthed to satisfy some SNR ratio
  target <- signal
  # snr <- 1
  # k <- sqrt(var(signal) / (var(noise) * snr))
  # target <- signal + k * noise
  
  # viz target values
  plot.ts(target[1:100])
  acf(target)
  acf(diff(target))
  
  # retrieve pertinent features, depending on chosen signal and noise types
  features <- get_features(y = target, signal_type = SIGNAL_TYPE, noise_type = NOISE_TYPE)
  
  # viz features
  str(features)
  head(features)
  
  # build proper dataframe to work with
  mat <- cbind.data.frame(target, features)
  mat$contains_na <- rowSums(is.na(mat) > 0)
  mat$is_train <- TRUE
  mat$is_train[(floor(n/2)+1):n] <- FALSE
  mat_col_index_target <- 1
  mat_col_index_features <- 1 + 1:ncol(features)
  
  # erase NAs
  mat <- na.omit(mat)
  
  ### LINEAR MODELS ---------------------------------------------------------------------------------
  
  # find \theta^*_0 := argmin_{\theta} L_0(Y, X\theta), where
  # L_0(y, y^) is the L2-norm of vector y-y^
  opt_L0_res <- optim_L0(y = mat[mat$is_train, mat_col_index_target], x = mat[mat$is_train, mat_col_index_features])
  agnostic_lm_param <- opt_L0_res$theta_hat
  
  # define grid search controls for optimization of L1 metric
  # also search controls
  theta_search_grid_controls_scenario1 <- list(
    'p' = length(mat_col_index_features),
    'lower_bound' = c(0, rep(-200, length(mat_col_index_features)-1)),
    'upper_bound' = c(500, rep(+200, length(mat_col_index_features)-1)),
    'nb_points' = rep(100, length(mat_col_index_features))
  )
  theta_search_grid_controls_scenario2 <- list(
    'p' = length(mat_col_index_features),
    'lower_bound' = c(0, rep(-1, length(mat_col_index_features)-1)),
    'upper_bound' = c(500, rep(+1, length(mat_col_index_features)-1)),
    'nb_points' = rep(100, length(mat_col_index_features))
  )
  theta_search_grid_controls <- theta_search_grid_controls_scenario2
  
  # direct gradient optimization
  opt_L1_res <- optim_grad_L1(y = mat[mat$is_train, mat_col_index_target], x = mat[mat$is_train, mat_col_index_features], 
                              lb = theta_search_grid_controls$lower_bound, up = theta_search_grid_controls$upper_bound, 
                              setup_pars = problem_constants, init_pars = opt_L0_res$theta_hat)
  goal_driven_lm_param <- opt_L1_res$par
  
  ### RANDOM FORESTS ---------------------------------------------------------------------------------
  
  source('custom_metrics.R')
  custom_loss <- function(y, y_pred){
    eval_theta(D = y, Dhat = y_pred)
  }
  
  # random forest hyper-parameters
  N_TREES <- 50
  MAX_DEPTH <- 6
  
  # fit single CART tree
  library(rpart)
  agnostic_rpart <- rpart(target ~ ., data = mat[mat$is_train, c(mat_col_index_target, mat_col_index_features)], control = rpart.control(maxdepth = MAX_DEPTH))
  
  # fit agnostic random forest
  library(randomForest)
  agnostic_rf <- randomForest(target ~ ., data = mat[mat$is_train, c(mat_col_index_target, mat_col_index_features)], mtry = 2, ntree = N_TREES)
  agnostic_rf_constrained <- randomForest(target ~ ., data = mat[mat$is_train, c(mat_col_index_target, mat_col_index_features)], mtry = 2, ntree = N_TREES, nodesize = 50, maxnodes = 70)
  agnostic_rf
  # compute_R2 <- function(y, yhat){
  #   1 - var(y-yhat) / var(y)
  # }
  # compute_R2(y = mat$target[mat$is_train], yhat = predict(agnostic_rf, mat[mat$is_train,]))
  # compute_R2(y = mat$target[mat$is_train==F], yhat = predict(agnostic_rf, mat[mat$is_train==F,]))
  # plot(mat$target[mat$is_train], predict(agnostic_rf, mat[mat$is_train,]))
  
  source('my_custom_rf.R')
  
  # redo predictions based on initial partition proposed by classic variance splitting
  my_forest <- recalibrate_forest_predictions(rfobj = agnostic_rf, 
                                              x_train = mat[mat$is_train, mat_col_index_features], 
                                              y_train = mat[mat$is_train, mat_col_index_target], 
                                              customized_loss_foo = custom_loss)
  my_forest_constrained <- recalibrate_forest_predictions(rfobj = agnostic_rf_constrained, 
                                                          x_train = mat[mat$is_train, mat_col_index_features], 
                                                          y_train = mat[mat$is_train, mat_col_index_target], 
                                                          customized_loss_foo = custom_loss)
  
  # fit customized random forest
  goal_driven_rf <- build_rf(y = mat[mat$is_train, mat_col_index_target],
                             x = mat[mat$is_train, mat_col_index_features],
                             customized_loss_foo = custom_loss, nb_points = 25, nb_points_y = 500, min_data_size = 50,
                             n_trees = 50, max_depth = MAX_DEPTH, bootstrap_prop = 0.632)
  
  # check that we have the right number of trees built
  length(goal_driven_rf)
  # look at second tree built
  str(goal_driven_rf[[1]], 1)
  # table(goal_driven_rf[[1]]$input_data_leaf_vec)
  # table(goal_driven_rf[[1]]$last_leaf_index)
  # View(goal_driven_rf[[1]]$tree_structure)
  
  ### EVALUATION ON THE TEST SET ---------------------------------------------------------------------------------------------
  
  #
  agnostic_rpart_pred <- predict(agnostic_rpart, mat[mat$is_train == FALSE, mat_col_index_features])
  
  # randomForest baseline, interesting to amke comparisons
  agnostic_rf_pred <- predict(agnostic_rf, mat[mat$is_train == FALSE, mat_col_index_features])
  agnostic_rf_constrained_pred <- predict(agnostic_rf_constrained, mat[mat$is_train == FALSE, mat_col_index_features])
  
  # recalibrated randomForest, compute predictions for each tree and aggregate with mean
  agnostic_to_goal_rf_pred <- predict_from_new_forest(myforest = my_forest, 
                                                      x_test = mat[mat$is_train == F, mat_col_index_features]) 
  agnostic_to_goal_rf_pred_mean <- rowMeans(agnostic_to_goal_rf_pred)
  agnostic_to_goal_rf_constrained_pred <- predict_from_new_forest(myforest = my_forest_constrained, 
                                                                  x_test = mat[mat$is_train == F, mat_col_index_features]) 
  agnostic_to_goal_rf_constrained_pred_mean <- rowMeans(agnostic_to_goal_rf_constrained_pred)
  
  # goal driven splits, but classic prediction metric
  goal_driven_rf_classic_pred <- t(apply(X = mat[mat$is_train == FALSE, mat_col_index_features], MARGIN = 1, FUN = function(x){
    predict_from_rf(rf = goal_driven_rf, x_vector = x, nb_points_y = 50, customized_loss_foo = sum_squared_errors)
  }))
  goal_driven_rf_classic_pred_mean <- rowMeans(goal_driven_rf_classic_pred)
  sd(goal_driven_rf_classic_pred)
  
  # goal driven splits and goal driven prediction
  goal_driven_rf_pred <- t(apply(X = mat[mat$is_train == FALSE, mat_col_index_features], MARGIN = 1, FUN = function(x){
    predict_from_rf(rf = goal_driven_rf, x_vector = x, nb_points_y = 50, customized_loss_foo = custom_loss)
  }))
  goal_driven_rf_pred_mean <- rowMeans(goal_driven_rf_pred)
  sd(goal_driven_rf_pred)
  ## apply(goal_driven_rf_pred, 1, sd)
  ## head(goal_driven_rf_pred)
  
  # agnostic linear model prediction
  agnostic_lm_pred <- as.numeric( as.matrix( mat[mat$is_train == FALSE, mat_col_index_features] ) %*% agnostic_lm_param )
  
  # task driven linear model prediction
  goal_driven_lm_pred <- as.numeric( as.matrix( mat[mat$is_train == FALSE, mat_col_index_features] ) %*% goal_driven_lm_param )
  
  # combine all the predictions in a big dataframe
  ypred_test <- cbind.data.frame(
    "oracle" = mat[mat$is_train == F, mat_col_index_target],
    "agnostic_lm" = agnostic_lm_pred,
    "goal_driven_lm" = goal_driven_lm_pred,
    "agnostic_rpart" = agnostic_rpart_pred,
    "agnostic_rf" = agnostic_rf_pred, #  + 30,
    "agnostic_to_goal_rf" = agnostic_to_goal_rf_pred_mean,
    "goal_driven_rf_classic" = goal_driven_rf_classic_pred_mean,
    "goal_driven_rf" = goal_driven_rf_pred_mean,
    "agnostic_rf_constr" = agnostic_rf_constrained_pred,
    "agnostic_to_goal_rf_constr" = agnostic_to_goal_rf_constrained_pred_mean
  )
  
  # correlation matrix between predictions -> it may be perfectly correlated, and yet there is something to gain - see below !
  cor(ypred_test)
  
  # plot test set (pred vs reality)
  color_vector <- rainbow(n = ncol(ypred_test))
  plot(x = ypred_test$oracle, y = ypred_test$oracle)
  for(pred_index in 2:ncol(ypred_test)){
    points(x = ypred_test$oracle, y = ypred_test[,pred_index], col = color_vector[pred_index])
  }
  legend("bottom", legend = colnames(ypred_test), col = color_vector, lty = 1, lwd = 2)
  
  # nicer plot
  library(ggplot2)
  temp <- data.table::melt(ypred_test, id.vars = "oracle")
  temp$rowname_Label <- ""
  temp$rowname_Label[temp$variable == 'agnostic_rpart'] <- "Agnostic CART"
  temp$rowname_Label[temp$variable == 'goal_driven_lm'] <- "LM w/ task prediction"
  temp$rowname_Label[temp$variable == 'goal_driven_rf'] <- "RF w/ task split, task prediction"
  temp$rowname_Label[temp$variable == 'goal_driven_rf_classic'] <- "RF w/ task split, classic prediction"
  temp$rowname_Label[temp$variable == 'agnostic_to_goal_rf'] <- "RF w/ classic split, task prediction"
  temp$rowname_Label[temp$variable == 'agnostic_to_goal_rf_constr'] <- "RF w/ classic split, task prediction, constrained"
  temp$rowname_Label[temp$variable == 'agnostic_lm'] <- "Agnostic Linear Model"
  temp$rowname_Label[temp$variable == 'agnostic_rf'] <- "Agnostic Random Forest"
  temp$rowname_Label[temp$variable == 'agnostic_rf_constr'] <- "Agnostic Random Forest, constrained"
  temp$rowname_Label <- factor(x = temp$rowname_Label, levels = unique(temp$rowname_Label))
  ggplot(data = temp, mapping = aes(x = oracle, y = value, col = rowname_Label)) + 
    geom_point(alpha = 0.05)  +
    geom_abline(slope = 1, intercept = 0) +
    geom_smooth(method = 'loess', se = FALSE) +
    theme(text = element_text(size = 10), legend.text = element_text(size = 5),
          title = element_text(size = 15),
          legend.position = 'top') + 
    labs(col = 'Model', x = 'True value', y = 'Estimated value') +
    ggtitle('Estimated vs true value on test set, n = 2500')
  
  # compute losses
  rmse_loss <- numeric(ncol(ypred_test))
  names(rmse_loss) <- colnames(ypred_test)
  goal_loss <- rmse_loss
  for(pred_index in 1:ncol(ypred_test)){
    goal_loss[pred_index] <- eval_theta(D = ypred_test$oracle, Dhat = ypred_test[,pred_index])
    rmse_loss[pred_index] <- mean((ypred_test$oracle - ypred_test[,pred_index])^2)
  }
  
  # quick look
  sort(goal_loss)
  sort(rmse_loss)
  
  # raw numbers plots
  barplot(sort(goal_loss), col = 'blue')
  barplot(sort(rmse_loss), col = 'red')
  
  # normalized plots
  normalize <- function(x){ (x - min(x)) / (max(x) - min(x)) }
  scores <- cbind(goal_loss, rmse_loss)
  scores <- scores[order(scores[,1]),]
  
  #
  resss[[length(resss) + 1]] <- data.frame(scores[,1:2]) %>% rownames_to_column() %>% mutate(experiment = iter)
  
}

saveRDS(object = resss, file = 'app1_experiments_3july2020.RDS')
resss <- readRDS(file = 'app1_experiments_3july2020.RDS')

# lapply(resss, function(x){
#   (min(x$goal_loss) == x$goal_loss[1]) & (min(x$rmse_loss) == x$rmse_loss[1])
# })
# resss_changed <- lapply(resss, function(x){
#   x$goal_loss <- (x$goal_loss - x$goal_loss[1]) 
#   x$goal_loss <- x$goal_loss / max(x$goal_loss)
#   x$rmse_loss <- x$rmse_loss / max(x$rmse_loss)
#   x[-1,]
# })
resss_changed <- resss
global_resss <- do.call(rbind.data.frame, resss_changed)
global_resss$rowname_Label <- ""
global_resss$rowname_Label[global_resss$rowname == 'agnostic_rpart'] <- "Agnostic CART"
global_resss$rowname_Label[global_resss$rowname == 'goal_driven_lm'] <- "LM w/ task prediction"
global_resss$rowname_Label[global_resss$rowname == 'goal_driven_rf'] <- "RF w/ task split, task prediction"
global_resss$rowname_Label[global_resss$rowname == 'goal_driven_rf_classic'] <- "RF w/ task split, classic prediction"
global_resss$rowname_Label[global_resss$rowname == 'agnostic_to_goal_rf'] <- "RF w/ classic split, task prediction"
global_resss$rowname_Label[global_resss$rowname == 'agnostic_to_goal_rf_constr'] <- "RF w/ classic split, task prediction, constrained"
global_resss$rowname_Label[global_resss$rowname == 'agnostic_lm'] <- "Agnostic Linear Model"
global_resss$rowname_Label[global_resss$rowname == 'agnostic_rf'] <- "Agnostic Random Forest"
global_resss$rowname_Label[global_resss$rowname == 'agnostic_rf_constr'] <- "Agnostic Random Forest, constrained"
global_resss$rowname_Label <- factor(x = global_resss$rowname_Label, levels = unique(global_resss$rowname_Label))

# saveRDS(object = global_resss, file = 'global_res_experiments_18juin2020_2.RDS')
# global_resss <- readRDS(file = 'global_res_experiments_18juin2020_2.RDS')

ggplot(data = global_resss) + 
  geom_boxplot(aes(x = rowname_Label, y = goal_loss, fill = 'Task-driven')) + 
  geom_boxplot(aes(x = rowname_Label, y = rmse_loss, fill = 'RMSE')) + 
  theme(text = element_text(size = 10),
        title = element_text(size = 15),
        legend.position = 'top',
        axis.text.x = element_text(size = 10, angle = 45, margin = margin(t = 70))) + 
  labs(fill = 'Loss', x = 'Model', y = 'Loss value') +
  ggtitle('Normalized loss (RMSE / Task-specific) on test set, n = 2500,', 'for the forecasters built from different techniques.') 


global_resss2 <- global_resss %>% pivot_longer(cols = c('goal_loss', 'rmse_loss'))
global_resss2$name_label <- factor(x = global_resss2$name, levels = c('goal_loss', 'rmse_loss'), labels = c('Task-Driven', 'RMSE'))

global_resss2$rowname_Label <- ""
global_resss2$rowname_Label[global_resss2$rowname == 'oracle'] <- "Oracle"
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_rpart'] <- "CART \n - \n - "
global_resss2$rowname_Label[global_resss2$rowname == 'goal_driven_lm'] <- "LM \n Task pred \n - "
global_resss2$rowname_Label[global_resss2$rowname == 'goal_driven_rf'] <- "RF \n Task pred \n Task split"
global_resss2$rowname_Label[global_resss2$rowname == 'goal_driven_rf_classic'] <- "RF \n - \n Task split"
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_to_goal_rf'] <- "RF \n Task pred \n -"
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_to_goal_rf_constr'] <- "RF \n Task pred \n - \n constr."
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_lm'] <- "LM \n - \n - "
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_rf'] <- "RF \n - \n - "
global_resss2$rowname_Label[global_resss2$rowname == 'agnostic_rf_constr'] <- "RF \n - \n - \n constr."
global_resss2$rowname_Label <- factor(x = global_resss2$rowname_Label, levels = unique(global_resss2$rowname_Label))

method_id <- 1 + 1 * grepl(x = levels(global_resss2$rowname_Label), 'LM') + 2 * grepl(x = levels(global_resss2$rowname_Label), 'RF') + 3 * grepl(x = levels(global_resss2$rowname_Label), 'CART')
vect_color <- viridis::inferno(n = 6, begin = 0, end = 0.8)
vect_color <- vect_color[method_id] # c("#CC00FF", "#CC3D3D", "#4D7E4F", '#1A80C4')[method_id]
new_vect_color <- vect_color
names(new_vect_color) <- levels(global_resss2$rowname_Label)
u_vect_color <- unique(vect_color)
names(u_vect_color) <- c('Oracle', 'LM', 'RF', 'CART')
gg1 <- ggplot(data = global_resss2) + 
  geom_boxplot(aes(x = rowname_Label, y = value, fill = name_label)) + 
  theme(text = element_text(size = 10),
        title = element_text(size = 10),
        legend.position = 'top',
        axis.text.x = element_text(size = 10, angle = 0, margin = margin(t = 10), colour = new_vect_color)) + 
  # scale_colour_manual(name = 'blabla', values = new_vect_color, aesthetics = 'colour') +
  labs(fill = 'Loss', x = 'Model', y = 'Loss value', col = 'Model') +
  ggtitle('RMSE & Task-Driven losses on test set, n = 2500,', 'for the forecasters built from different techniques.') +
  # ggtitle('Normalized loss (RMSE / Task-Driven) on test set, n = 2500,', 'for the forecasters built from different techniques.') +
  facet_grid(name_label~., scales = 'free')
pdf(file = 'raw_losses_task_driven_approach.pdf')
gg1
dev.off()



par(mfrow = c(2,1))
barplot(normalize(scores[,1]), col = 'blue', main = "Task loss")
barplot(normalize(scores[,2]), col = 'red', main = "RMSE")
par(mfrow = c(1,1))

#
normalized_scores <- scores
for(j in 1:ncol(scores)) normalized_scores[,j] <- normalize(normalized_scores[,j])
normalized_scores <- data.table::melt(normalized_scores)
normalized_scores$Var1_Label <- ""
normalized_scores$Var1_Label[normalized_scores$Var1 == 'goal_driven_lm'] <- "LM w/ task prediction"
normalized_scores$Var1_Label[normalized_scores$Var1 == 'goal_driven_rf'] <- "RF w/ task split, task prediction"
normalized_scores$Var1_Label[normalized_scores$Var1 == 'goal_driven_rf_classic'] <- "RF w/ task split, classic prediction"
normalized_scores$Var1_Label[normalized_scores$Var1 == 'agnostic_to_goal_rf'] <- "RF w/ classic split, task prediction"
normalized_scores$Var1_Label[normalized_scores$Var1 == 'agnostic_lm'] <- "Agnostic Linear Model"
normalized_scores$Var1_Label[normalized_scores$Var1 == 'agnostic_rf'] <- "Agnostic Random Forest"
normalized_scores$Var2_Label <- "RMSE"
normalized_scores$Var2_Label[normalized_scores$Var2 == 'goal_loss'] <- "Task"
normalized_scores$Var1_Label <- factor(x = normalized_scores$Var1_Label, levels = unique(normalized_scores$Var1_Label))
library(ggplot2)
ggplot(data = normalized_scores[!normalized_scores$Var1 %in% c('oracle', 'agnostic_rpart'),]) +
  geom_bar(mapping = aes(x = Var1_Label, y = value, fill = factor(Var2_Label)), stat = 'identity', position = 'dodge') +
  ggtitle('Normalized loss (RMSE / Task-specific) on test set', 'for the forecasters built from different techniques.') +
  theme(
    title = element_text(size = 20), # , margin = ggplot2::margin(t = 20, 0, 0, b = 20)),
    legend.text = element_text(size = 20),
    axis.text.y = element_text(size = 15),
    axis.text.x = element_text(size = 15, angle = 45, margin = margin(t = 70))
  ) + 
  labs(fill = 'Loss') +
  xlab('') +
  ylab('') 



