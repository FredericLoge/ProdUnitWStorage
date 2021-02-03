bingaloo_1 <- readRDS(file = 'model_1_policy.RDS')
bingaloo_2 <- readRDS(file = 'model_2_policy.RDS')

bingaloo_1$production_level <- factor(bingaloo_1$production_level, levels = seq(0, 1,  by = 0.25))
bingaloo_1$storage_change <- factor(bingaloo_1$storage_change, levels = c(-0.1, 0, 0.1))
bingaloo_2$production_level <- factor(bingaloo_2$production_level, levels = seq(0, 1,  by = 0.25))
bingaloo_2$storage_change <- factor(bingaloo_2$storage_change, levels = c(-0.1, 0, 0.1))

# head(bingaloo_1)
# bingaloo_1$demand <- rep(seq(60, 400, length.out = 50), times = 50)
# bingaloo_2$demand <- rep(seq(60, 400, length.out = 50), times = 50)
# range(bingaloo_1$storage)
# range(bingaloo_2$storage)
# bingaloo_1$model <- "Model 1"
# bingaloo_2$model <- "Model 2"
# bingaloo_big <- rbind.data.frame(bingaloo_1, bingaloo_2)
# bingaloo_big$model <- factor(bingaloo_big$model)
# bingaloo_big$production_level <- factor(bingaloo_big$production_level)
# bingaloo_big$storage_change <- factor(bingaloo_big$storage_change)
# bingaloo_big$demand <- as.numeric(as.character(factor(bingaloo_big$demand)))
# bingaloo_big$storage <- as.numeric(as.character(factor(bingaloo_big$storage)))

# library(ggplot2)
# graph_opt_prod <- ggplot(data = bingaloo_big) +
#   aes(x = demand, y = storage, fill = production_level) +
#   geom_tile() +
#   scale_fill_viridis_d() +
#   facet_grid(.~model) +
#   labs(fill = 'Production level', x = 'Current customer demand Y[t]', y = 'Current storage level')
# graph_opt_storage <- ggplot(data = bingaloo_big) +
#   aes(x = demand, y = storage, fill = storage_change) +
#   geom_tile() +
#   scale_fill_viridis_d() +
#   facet_grid(.~model) +
#   labs(fill = 'Storage decision', x = 'Prior customer demand Y[t-1]', y = 'Current storage level')
# library(gridExtra)
# gridExtra::grid.arrange(graph_opt_prod, graph_opt_storage, ncol = 1, nrow = 2, top = 'Production and storage decisions based on current state')

library(ggplot2)
graph_opt_prod_1 <- ggplot(data = bingaloo_1) +
  aes(x = demand, y = storage, fill = (production_level)) +
  geom_tile() +
  scale_fill_viridis_d(drop = FALSE) +
  theme(legend.position = 'top') +
  labs(fill = 'Production level', x = expression(paste("Current customer demand ", Y[t])), y = 'Current storage level')
graph_opt_storage_1 <- ggplot(data = bingaloo_1) +
  aes(x = demand, y = storage, fill = factor(storage_change)) +
  geom_tile() +
  scale_fill_viridis_d(drop = FALSE) +
  theme(legend.position = 'top') +
  labs(fill = 'Storage decision', x = expression(paste("Current customer demand ", Y[t])), y = 'Current storage level')
graph_opt_prod_2 <- ggplot(data = bingaloo_2) +
  aes(x = demand, y = storage, fill = production_level) +
  geom_tile() +
  scale_fill_viridis_d(drop = FALSE) +
  theme(legend.position = 'top') +
  labs(fill = 'Production level', x = expression(paste("Prior customer demand ", Y[t-1])), y = 'Current storage level')
graph_opt_storage_2 <- ggplot(data = bingaloo_2) +
  aes(x = demand, y = storage, fill = (storage_change)) +
  geom_tile() +
  scale_fill_viridis_d(drop = FALSE) +
  theme(legend.position = 'top') +
  labs(fill = 'Storage decision', x = expression(paste("Prior customer demand ", Y[t-1])), y = 'Current storage level')

library(gridExtra)
pdf(file = 'model_1_with_storage.pdf', height = 7, width = 4.5)
gridExtra::grid.arrange(graph_opt_prod_1, 
                        graph_opt_storage_1,
                        nrow = 2, ncol = 1,
                        top = 'Model 1 computed from data')
dev.off()
pdf(file = 'model_2_with_storage.pdf', height = 7, width = 4.5)
gridExtra::grid.arrange(graph_opt_prod_2, 
                        graph_opt_storage_2,
                        nrow = 2, ncol = 1,
                        top = 'Model 2 computed from data')
dev.off()

# 
# rev(unique(bingaloo_1$demand))[1:10]
# rev(unique(bingaloo_2$demand))[1:10]
blabou <- readRDS(file = 'model_1_data_generated.RDS')
plot.ts(blabou[1:30,])
