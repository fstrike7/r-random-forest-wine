# Random Forest - Clasificación de Calidad de Vinos

# Librerias
library(tidyverse)
library(caret)
library(randomForest)
library(pROC)

if (!dir.exists("img")) dir.create("img")

# Dataset
wine <- read.csv("data/winequality-red.csv", sep = ";")

wine$quality_label <- ifelse(wine$quality >= 7, "Alta", "Baja")
wine$quality_label <- factor(wine$quality_label, levels = c("Baja", "Alta"))

# train/test
set.seed(123) # opcional: para replicar resultados.
split <- createDataPartition(wine$quality_label, p = 0.8, list = FALSE)
train_data <- wine[split, ]
test_data <- wine[-split, ]

# entreno
set.seed(123)
model_rf <- randomForest(quality_label ~ . -quality, data = train_data, importance = TRUE, ntree = 500)

# predicciones
pred_rf <- predict(model_rf, newdata = test_data)

# matriz de confusión y métricas
conf_mat <- confusionMatrix(pred_rf, test_data$quality_label)
print(conf_mat)

# exportar gráficos
conf_df <- as.data.frame(conf_mat$table)
conf_plot <- ggplot(conf_df, aes(Prediction, Reference, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), color = "white", size = 6) +
  scale_fill_gradient(low = "steelblue", high = "darkred") +
  labs(title = "Matriz de Confusión - Random Forest", x = "Predicción", y = "Real") +
  theme_minimal()

ggsave("img/confusion_matrix.png", conf_plot, width = 6, height = 4)

# feature_importance -> importancia de cada variable
png("img/feature_importance.png", width = 800, height = 600)
varImpPlot(model_rf, main = "Importancia de Variables")
dev.off()

# AUC - eficacia del modelo
prob_rf <- predict(model_rf, newdata = test_data, type = "prob")[, "Alta"]
roc_obj <- roc(test_data$quality_label, prob_rf)
cat("AUC:", auc(roc_obj), "\n")
