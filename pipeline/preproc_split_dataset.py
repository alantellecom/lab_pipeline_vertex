@component(
    packages_to_install = [
        "pandas",
        "sklearn==0.20.4"
    ],
)


def preproc_split_dataset(dataset_train: Output[Dataset],dataset_validation: Output[Dataset], dataset_test: Output[Dataset]):

  import pandas as pd
  from sklearn.model_selection import train_test_split
  from sklearn import preprocessing
  import os

  os.system('curl -o iris.txt  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')  
  os.system('gsutil cp  iris.txt  {}'.format(root_path))
    
    
  columns = ['sepal_length','sepal_width','petal_length','petal_width','class']

  dfIris = pd.read_csv('{}/iris.txt'.format(root_path), names=columns)
  le = preprocessing.LabelEncoder()
  dfIris['classEncoder'] = le.fit_transform(dfIris['class'])
    
  X_train, X_val_test, y_train, y_val_test = train_test_split(dfIris.drop(['classEncoder','class'], axis=1), dfIris['classEncoder'], test_size=0.2)

  X_val, X_test, y_val, y_test = train_test_split(X_val_test,y_val_test, test_size=0.5)
    
  df_train = pd.concat([X_train,y_train],axis=1)
  df_val = pd.concat([X_val,y_val],axis=1)
  df_test = pd.concat([X_test,y_test],axis=1)
  
  df_train.to_csv(dataset_train.path, index=False)
  df_val.to_csv(dataset_validation.path, index=False)
  df_test.to_csv(dataset_test.path , index=False)