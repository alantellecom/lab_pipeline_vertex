from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        component)
@component(
    packages_to_install = [
        "pandas",
        "scikit-learn==0.20.4",
        "google-cloud-storage",
        "gcsfs",
        "fsspec"
    ],
)

def evaluate_model(
    test_set: Input[Dataset],
    model_artifact: Input[Model],
    smetrics: Output[Metrics]
):

  import pickle
  import pandas as pd
  from sklearn.metrics import accuracy_score, recall_score
  
  from google.cloud import storage

  df_test = pd.read_csv(test_set.path)

  X_test = df_test.drop('classEncoder', axis=1)
  y_test = df_test['classEncoder']

  model_filename = 'model.pkl' 
  models_artifact_path = model_artifact.path
  models_artifact_path=models_artifact_path.replace('/gcs/','gs://')
  model_path = '{}/{}'.format(models_artifact_path,model_filename)
  
  blob = storage.blob.Blob.from_string(model_path, client=storage.Client())
  blob.download_to_filename(model_filename)
  
  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

  y_hat = model.predict(X_test)
  metric_value = accuracy_score(y_test, y_hat)
            
  smetrics.log_metric("score", float(metric_value))