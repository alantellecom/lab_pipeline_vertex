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

def train_model(
    dataset_train: Input[Dataset],
    dataset_validation: Input[Dataset],
    best_hp_values: Input[Model],
    model_artifact: Output[Model]
):
  import pickle
  import os

  import pandas as pd
  from sklearn.linear_model import SGDClassifier
  from sklearn.pipeline import Pipeline

  from google.cloud import storage

  df_train = pd.read_csv(dataset_train.path)
  df_validation = pd.read_csv(dataset_validation.path)

  df_train = pd.concat([df_train, df_validation])

  pipeline = Pipeline([('classifier', SGDClassifier(loss='log'))])

  print('Starting training: alpha={}, max_iter={}'.format(best_hp_values.metadata["best_learning_rate"], best_hp_values.metadata["best_iteration"]))
  X_train = df_train.drop('classEncoder', axis=1)
  y_train = df_train['classEncoder']

  pipeline.set_params(classifier__alpha=best_hp_values.metadata["best_learning_rate"], 
                      classifier__max_iter=best_hp_values.metadata["best_iteration"])
  pipeline.fit(X_train, y_train)
  model_artifact.metadata["train_score"]=float(pipeline.score(X_train, y_train))
  model_artifact.metadata["framework"] = "sklearn"
        
  model_filename = 'model.pkl'
  local_path = model_filename 
  
  with open(local_path , 'wb') as model_file:
    pickle.dump(pipeline, model_file)

  models_path = model_artifact.path
  models_path=models_path.replace('/gcs/','gs://')
  
  blob = storage.blob.Blob.from_string('{}/{}'.format(models_path,model_filename), client=storage.Client())
  blob.upload_from_filename(model_filename)
    