bingaloo <- readRDS(file = 'model_1_policy.RDS')
initial_state_df <- readRDS(file = 'model_1_data_generated.RDS')
summary(initial_state_df)

# bingaloo <- readRDS(file = 'model_2_policy.RDS')
# initial_state_df <- readRDS(file = 'model_2_data_generated.RDS')

### VISUALIZE POLICY ====

# wireframe
library(lattice)
wireframe(production_level ~ demand * storage, data = bingaloo,
          scales = list(arrows = FALSE),
          drape = TRUE, colorkey = TRUE,
          screen = list(z = 30, x = -60))
wireframe(storage_change ~ demand * storage, data = bingaloo,
          scales = list(arrows = FALSE),
          drape = TRUE, colorkey = TRUE,
          screen = list(z = 30, x = -60))

# ggplots
library(ggplot2)
graph_opt_prod <- ggplot(data = bingaloo) +
  aes(x = demand, y = storage, fill = production_level) +
  geom_tile() +
  scale_fill_viridis_c()
graph_opt_storage <- ggplot(data = bingaloo) +
  aes(x = demand, y = storage, fill = storage_change) +
  geom_tile() +
  scale_fill_viridis_c()
## initial_state_df <- data.frame(initial_state)
some_data_points <- ggplot(data = initial_state_df) +
  aes(x = state_demand, y = state_storage) +
  geom_hex() +
  scale_fill_viridis_c()

library(gridExtra)
gridExtra::grid.arrange(graph_opt_prod, graph_opt_storage, some_data_points, layout_matrix = matrix(c(1, NA, 2, 3), byrow = TRUE, ncol = 2))
