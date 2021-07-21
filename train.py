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
    alpha: float,
    max_iter: int,
    model_artifact: Output[Model],
    root_path:str
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

  print('Starting training: alpha={}, max_iter={}'.format(alpha, max_iter))
  X_train = df_train.drop('classEncoder', axis=1)
  y_train = df_train['classEncoder']

  pipeline.set_params(classifier__alpha=alpha, classifier__max_iter=max_iter)
  pipeline.fit(X_train, y_train)
  model_artifact.metadata["train_score"]=float(pipeline.score(X_train, y_train))
  model_artifact.metadata["framework"] = "sklearn"

  model_filename = 'model.pkl'
  local_path = model_filename 
  with open(local_path , 'wb') as model_file:
    pickle.dump(pipeline, model_file)
    blob = storage.blob.Blob.from_string('{}/models/{}'.format(root_path,model_filename), client=storage.Client())
    blob.upload_from_filename(model_filename)
    