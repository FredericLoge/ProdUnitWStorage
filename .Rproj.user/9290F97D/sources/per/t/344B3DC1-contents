# choose oracle (not) and load appropriate policy
model_oracle = TRUE
if(model_oracle){
  model_policy <- readRDS(file = 'model_1_policy.RDS')
  model_policy <- unique(readRDS(file = 'model_1_policy_nostorage.RDS'))
}else{
  model_policy <- readRDS(file = 'model_2_policy.RDS')
  model_policy <- unique(readRDS(file = 'model_2_policy_nostorage.RDS'))
}

#
set.seed(2852)
NN_TEST <- 1000
demand_sequence_TEST <- sim_ar_bis(n = NN_TEST + 1)
initial_storage_TEST <- c(runif(n = 1, min = MIN_STORAGE, max = MAX_STORAGE), rep(NA, NN_TEST - 1))
test_df <- data.frame(
  'past_demand' = demand_sequence_TEST[- (NN_TEST + 1)],
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
my_fooo(demand = 200, storage = 0, bingaloo = model_policy)

for(i in 1:NN_TEST){
  
  # compute decision and store it in dataframe
  if(model_oracle == TRUE){
    my_decision <- my_fooo(demand = test_df$demand[i], storage = test_df$storage[i], bingaloo = model_policy)
  }else{
    my_decision <- my_fooo(demand = test_df$past_demand[i], storage = test_df$storage[i], bingaloo = model_policy)
  }
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

colnames(test_df)

pdf(file = 'illustration_time_series_system.pdf')
plot.ts(test_df[20 + 1:40, c(2, 3, 8, 9, 10, 11)], type = 'o', pch = 20, main = 'Illustrative example of the system following model 1 \n with perfect information')
dev.off()

library(ggplot2)
test_df$time <- 1:nrow(test_df)
ggplot(data = test_df[20 + 1:40,]) +
  geom_line(aes(x = time, y = demand, col = 'demand'), lwd = 1, pch = 20, cex = 2) +
  geom_line(aes(x = time, y = Supplyt, col = 'Supplyt'), lwd = 1) 
ggplot(data = test_df[20 + 1:40,]) +
  geom_line(aes(x = time, y = REWARD, col = 'REWARD'), lwd = 1, pch = 20, cex = 2) 

#
#
test_info <- list(
  'summary' = summary(test_df$REWARD)
)

## Oracle (no storage)
#  Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3200   -1997   -1292   -1183    -369     800

# Oracle ===
#    Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3377.0 -2012.2 -1340.2 -1266.5  -426.7   733.9
#
# NNET 2
#     Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
# -3261.2 -1323.2   128.6   111.7  1433.7  8098.5 
plot.ts(test_df$storage)
plot.ts(test_df$decision_storage_change)
plot.ts(test_df$decision_production_level)

if(model_oracle){
  fn <- 'test_model_1_info.RDS'
}else{
  fn <- 'test_model_2_info.RDS'
}
saveRDS(object = test_info, file = fn)
