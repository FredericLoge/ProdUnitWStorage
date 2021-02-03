
### PARAMETERS ====

# set of known constants in equations (1), (2), (3) and constraints on \tau
problem_constants <- list(
  'a' = 300, # slope of (1)
  'b' = 100, # intercept of (1)
  'c' = 0.9, # production efficiency of our units (2)
  'd' = 10,  # unit cost of input energy (3)
  'e' = 100, # 100,  # unit cost / penalty due to lack of supply (3)
  'f' = 20,   # unit reward per production output
  'tau_min' = 0.0, # minimal functioning level
  'tau_max' = 1.0  # maximal functioning level
)
problem_constants$P_min <- with(problem_constants, c*(a*tau_min + b))
problem_constants$P_max <- with(problem_constants, c*(a*tau_max + b)) 
pars <- problem_constants
pars$P_min
pars$P_max

sim_ar_1 <- function(n, sigma2e, theta){
  
  # storing simulation in vec s
  s <- numeric(n)
  
  # get p from vec beta
  p <- length(theta)
  
  # simulate under AR(p)
  for(i in 2:n){
    s[i] <- s[i-1] * theta[1] + rnorm(n = 1, mean = 0, sd = sqrt(sigma2e))
  }
  
  # normalize  
  # range(s)
  # pars$P_max
  # pars$P_min
  s <- with(problem_constants, (s - mean(s)) / sd(s) * 8 * sqrt(P_max - P_min) + (P_max + P_min)/2)

  # return sample
  return(s)
  
}

## sim_ar_1 <- function(n, sigma2e, theta){
#   
#   # storing simulation in vec s
#   s <- numeric(n)
#   
#   # get p from vec beta
#   p <- length(theta)
#   
#   # simulate under AR(p)
#   for(i in 2:n){
#     s[i] <- s[i-1] * theta[1] + rnorm(n = 1, mean = 0, sd = sqrt(sigma2e))
#   }
#   
#   # normalize  
#   s_range <- sqrt( 2 * log(2 * n / 0.01) / (1 - theta^2))
#   s <- with(problem_constants, P_min + (s + s_range) / (2 * s_range) * (P_max - P_min) * 3)
#   
#   # return sample
#   return(s)
#   
# }


# demand range
demand_sequence_TEST <- sim_ar_1(n = 10000, sigma2e = 0.5, theta = 0.3)
quantile(demand_sequence_TEST, probs = c(0.05, 0.95))
quantile(demand_sequence_TEST, probs = c(0.1, 0.9))
MAX_DEMAND <- 400
MIN_DEMAND <- 60
demand_sequence_TEST <- pmax(pmin(demand_sequence_TEST, MAX_DEMAND), MIN_DEMAND)

sim_ar_bis <- function(n){
  simu <- sim_ar_1(n = n, sigma2e = 0.5, theta = 0.3)  
  simu <- pmax(pmin(simu, MAX_DEMAND), MIN_DEMAND)
  return(simu)
}

# storage range
MAX_STORAGE <- 200
MIN_STORAGE <- 0

# storage range (go to 0)
MAX_STORAGE <- 0
MIN_STORAGE <- 0
