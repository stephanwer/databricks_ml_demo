# Databricks notebook source
# MAGIC %run  ./../utils/config

# COMMAND ----------

# MAGIC %run ./../data/data_transformations

# COMMAND ----------


model_name = "ml-model-demo"
env = 'dev'
experiment_id = env_experiment_id_dict['dev']

client = mlflow.tracking.MlflowClient()
production_model = client.get_latest_versions(name = model_name, stages = ["Production"])[0]
DataProvider = LendingClubDataProvider(spark)

# COMMAND ----------


_, X_test, _, _ = DataProvider.run()
model = mlflow.pyfunc.load_model(production_model.source)
df_predictions = model.predict(X_test)

# COMMAND ----------

sdf_predictions = spark.createDataFrame(pd.DataFrame(df_predictions), "prediction: int")
display(sdf_predictions)
