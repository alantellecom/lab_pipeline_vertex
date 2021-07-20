@component(
    packages_to_install = [
        "pandas",
        "scikit-learn==0.20.4",
        "pickle"
    ],
)

import pickle
import os

import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline


def train_model(
    dataset_train: Input[Dataset],
    dataset_validation: Input[Dataset],
    alpha,
    max_iter,
    model_artifact: Output[Model]
):

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
  # Upload model artifact to Cloud Storage
    model_directory = os.environ['AIP_MODEL_DIR']
    storage_path = os.path.join(model_directory, model_filename)
    os.system('gsutil cp {} {}'.format(local_path,storage_path)

  

