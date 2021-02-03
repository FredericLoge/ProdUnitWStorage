# READ INFOS ====

bingaloo <- readRDS(file = 'model_1_policy.RDS')
initial_state_df <- readRDS(file = 'model_1_data_generated.RDS')

my_bingaloo <- unique(readRDS(file = 'model_1_policy_nostorage.RDS'))
initial_state_df <- readRDS(file = 'model_1_data_generated_nostorage.RDS')

# CLASSIC LINEAR MODEL ====

# classic linear model
my_lm <- lm(next_state_demand ~ state_demand, data = initial_state_df)

# test linear model

set.seed(2852)

NN_TEST <- 1000
demand_sequence_TEST <- sim_ar_bis(n = NN_TEST + 1)
plot.ts(demand_sequence_TEST)
lmdf <- data.frame(
  y = demand_sequence_TEST[-1],
  x = demand_sequence_TEST[-(NN_TEST+1)]
)
str(lmdf)
newlm <- (lm(y~x, data=lmdf))
coefficients(newlm)
sd(newlm$residuals)
hist(newlm$residuals)

initial_storage_TEST <- c(runif(n = 1, min = MIN_STORAGE, max = MAX_STORAGE), rep(NA, NN_TEST - 1))
test_df <- data.frame(
  'past_demand' = demand_sequence_TEST[- (NN_TEST + 1)],
  'predicted_demand' = coefficients(my_lm)[1] + coefficients(my_lm)[2] * demand_sequence_TEST[- (NN_TEST + 1)],
  'demand' = demand_sequence_TEST[- 1],
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
my_fooo <- function(demand, storage, bingaloo){
  valeur_plus_proche_demand <- bingaloo$demand[which.min(abs(bingaloo$demand - demand))]
  valeur_plus_proche_storage <- bingaloo$storage[which.min(abs(bingaloo$storage - storage))]
  indexx <- which( (abs(bingaloo$demand - valeur_plus_proche_demand) < 1e-10) & (abs(bingaloo$storage - valeur_plus_proche_storage) < 1e-10) )
  c("storage_change" = bingaloo$storage_change[indexx],
    "production_level" = bingaloo$production_level[indexx])
}
my_fooo(demand = 200, storage = 0, bingaloo = my_bingaloo)

for(i in 1:NN_TEST){
  
  # compute decision and store it in dataframe
  my_decision <- my_fooo(demand = test_df$predicted_demand[i], storage = test_df$storage[i], bingaloo = my_bingaloo)
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

# la marat des bois
# ...

agnostic_lm_info <- list(
  'lm_coefficients' = coefficients(my_lm),
  'summary' = summary(test_df$REWARD)
)
saveRDS(object = agnostic_lm_info, file = 'agnostic_lm_info.RDS')


# TASK-DRIVEN LINEAR MODEL ====

# test linear model
search_space <- expand.grid(
  'intercept' = seq(from = coefficients(my_lm)[1] * 0.5, to = coefficients(my_lm)[1] * 1.5, length.out = 10),
  'slope' = seq(from = coefficients(my_lm)[2] * 0.5, to = coefficients(my_lm)[2] * 1.5, length.out = 10)
)
search_space$mean_reward <- NA

for(index in 1:nrow(search_space)){
  
  cat('\n', index)
  
  set.seed(397643)
  NN_TEST <- 1000
  demand_sequence_TEST <- sim_ar_bis(n = NN_TEST + 1)
  initial_storage_TEST <- c(runif(n = 1, min = MIN_STORAGE, max = MAX_STORAGE), rep(NA, NN_TEST - 1))
  test_df <- data.frame(
    'past_demand' = demand_sequence_TEST[- (NN_TEST + 1)],
    'predicted_demand' = search_space$intercept[index] + search_space$slope[index] * demand_sequence_TEST[- (NN_TEST + 1)],
    'demand' = demand_sequence_TEST[- 1],
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
  my_fooo <- function(demand, storage, bingaloo){
    valeur_plus_proche_demand <- bingaloo$demand[which.min(abs(bingaloo$demand - demand))]
    valeur_plus_proche_storage <- bingaloo$storage[which.min(abs(bingaloo$storage - storage))]
    indexx <- which( (abs(bingaloo$demand - valeur_plus_proche_demand) < 1e-10) & (abs(bingaloo$storage - valeur_plus_proche_storage) < 1e-10) )
    c("storage_change" = bingaloo$storage_change[indexx],
      "production_level" = bingaloo$production_level[indexx])
  }
  my_fooo(demand = 200, storage = 0, bingaloo = my_bingaloo)
  
  for(i in 1:NN_TEST){
    
    # compute decision and store it in dataframe
    my_decision <- my_fooo(demand = test_df$predicted_demand[i], storage = test_df$storage[i], bingaloo = my_bingaloo)
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
  
  search_space$mean_reward[index] <- mean(test_df$REWARD, na.rm = TRUE)
  
}

#
final_index <- which.min(search_space$mean_reward)
search_space[final_index,]
coefficients(my_lm)

#
set.seed(2852)
NN_TEST <- 1000
demand_sequence_TEST <- sim_ar_bis(n = NN_TEST + 1)
initial_storage_TEST <- c(runif(n = 1, min = MIN_STORAGE, max = MAX_STORAGE), rep(NA, NN_TEST - 1))
test_df <- data.frame(
  'past_demand' = demand_sequence_TEST[- (NN_TEST + 1)],
  'predicted_demand' = search_space$intercept[final_index] + search_space$slope[final_index] * demand_sequence_TEST[- (NN_TEST + 1)],
  'demand' = demand_sequence_TEST[- 1],
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
my_fooo <- function(demand, storage, bingaloo){
  valeur_plus_proche_demand <- bingaloo$demand[which.min(abs(bingaloo$demand - demand))]
  valeur_plus_proche_storage <- bingaloo$storage[which.min(abs(bingaloo$storage - storage))]
  indexx <- which( (abs(bingaloo$demand - valeur_plus_proche_demand) < 1e-10) & (abs(bingaloo$storage - valeur_plus_proche_storage) < 1e-10) )
  c("storage_change" = bingaloo$storage_change[indexx],
    "production_level" = bingaloo$production_level[indexx])
}
my_fooo(demand = 200, storage = 0, bingaloo = my_bingaloo)

for(i in 1:NN_TEST){
  
  # compute decision and store it in dataframe
  my_decision <- my_fooo(demand = test_df$predicted_demand[i], storage = test_df$storage[i], bingaloo = my_bingaloo)
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


#
#
taskdriven_lm_info <- list(
  'lm_coefficients' = search_space[final_index,c(1,2)],
  'summary' = summary(test_df$REWARD)
)
saveRDS(object = taskdriven_lm_info, file = 'taskdriven_lm_info.RDS')

taskdriven_lm_info
agnostic_lm_info
test_info

## agnostic predictor (model 1)
#
#     Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -2652.0  -823.9   857.2  2426.6  4220.6 18200.0 
#
## task-driven predictor (model 1)
#
#     Min.  1st Qu.   Median     Mean  3rd Qu.     Max. 
# -3190.51 -1338.28    85.82   150.60  1424.73  8150.00 
# 
## perfect predictor (model 1)
#
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3377.0 -2012.2 -1340.2 -1266.5  -426.7   733.9
#
## global optim (model 2)
#
#     Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3261.2 -1323.2   128.6   111.7  1433.7  8098.5 

## NO STORAGE
## model 1 + task driven
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3190.5 -1338.3   125.9   133.6  1406.3  8150.0 
## model 1 + agnostic
#     Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -2599.5  -881.4   671.4  2116.6  3373.1 15500.0 
