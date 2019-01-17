library(mlflow)
library(sparkxgb)
library(sparklyr)

sc <- spark_connect(master = "local")
iris_tbl <- sdf_copy_to(sc, iris)

with(mlflow_start_run(), {
  num_trees <- mlflow_param("num_trees", 50)
  max_depth <- mlflow_param("max_depth", 4)
  
  rf_model <- ml_random_forest_classifier(
    iris_tbl, 
    Species ~ .,
    num_trees = num_trees, 
    max_depth = max_depth
  )
  
  rf_accuracy <- rf_model %>%
    ml_predict(iris_tbl) %>%
    ml_multiclass_classification_evaluator(metric_name = "accuracy")
  
  mlflow_log_param("num_trees", num_trees)
  mlflow_log_param("max_depth", max_depth)
  mlflow_set_tag("model_type", "random_forest")
  mlflow_log_metric("accuracy", rf_accuracy)
})

