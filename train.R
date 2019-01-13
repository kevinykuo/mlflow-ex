library(mlflow)
library(sparkxgb)
library(sparklyr)

sc <- spark_connect(master = "local")
iris_tbl <- sdf_copy_to(sc, iris)

with(mlflow_start_run(), {
  num_trees <- mlflow_param("num_trees", 50)
  max_depth <- mlflow_param("max_depth", 4)
  
  xgb_model <- xgboost_classifier(
    iris_tbl, 
    Species ~ .,
    objective = "multi:softprob",
    num_class = 3,
    num_round = num_trees, 
    max_depth = max_depth
  )
  
  sparkxgb_accuracy <- xgb_model %>%
    ml_predict(iris_tbl) %>%
    ml_multiclass_classification_evaluator(metric_name = "accuracy")
  
  mlflow_log_param("num_trees", num_trees)
  mlflow_log_param("max_depth", max_depth)
  mlflow_log_metric("accuracy", sparkxgb_accuracy)
})

