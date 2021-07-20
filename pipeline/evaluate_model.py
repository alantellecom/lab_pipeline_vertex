
@component(
    packages_to_install = [
        "pandas",
        "scikit-learn==0.20.4",
        "pickle"
    ],
)

def evaluate_model(
    test_set: Input[Dataset],
    model: Input[Model],
    smetrics: Output[Metrics]
):

  import pickle
  import pandas as pd
  from sklearn.metrics import accuracy_score, recall_score
  

  df_test = pd.read_csv(test_set.path)

  X_test = df_test.drop('classEncoder', axis=1)
  y_test = df_test['classEncoder']

  model_filename = 'model.pkl'
  local_path = model_filename 
  model_directory = os.environ['AIP_MODEL_DIR']
  storage_path = os.path.join(model_directory, model_filename)
  os.system('gsutil cp {} {}'.format(storage_path,local_path)

  with open(model_filename, 'rb') as model_file:
    model = pickle.load(model_file)

  y_hat = model.predict(X_test)
  metric_value = accuracy_score(y_test, y_hat)
            
  smetrics.log_metric("score", float(metric_value))