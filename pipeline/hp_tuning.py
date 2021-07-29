from kfp.v2.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        component)

@component(
    packages_to_install = [
        "google-cloud-aiplatform"
    ],
)


def hyperparameter_tuning_job(
    dataset_train: Input[Dataset],
    dataset_validation: Input[Dataset],
    project: str,
    display_name: str,
    image_uri: str,
    best_hp_values: Output[Model],
    #package_uri: str,
    #python_module: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com"
):
    from google.cloud import aiplatform
    import time
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.JobServiceClient(client_options=client_options)

    # study_spec
    metric = {
        "metric_id": "accuracy",
        "goal": aiplatform.gapic.StudySpec.MetricSpec.GoalType.MAXIMIZE,
    }

    parameter_learning_rate = {
            "parameter_id": "learning_rate",
            "double_value_spec": {"min_value": 1e-02, "max_value": 4e-02},
            "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
    }
    parameter_iterations = {
        "parameter_id": "iteration",
        "integer_value_spec": {"min_value": 70, "max_value": 75},
        "scale_type": aiplatform.gapic.StudySpec.ParameterSpec.ScaleType.UNIT_LINEAR_SCALE,
    }

    # trial_job_spec
    machine_spec = {
        "machine_type": "n1-standard-4",
        #"accelerator_type": aiplatform.gapic.AcceleratorType.NVIDIA_TESLA_K80,
        #"accelerator_count": 1,
    }
    worker_pool_spec = {
        "machine_spec": machine_spec,
        "replica_count": 1,
         "container_spec":{
             "image_uri": image_uri,
             "args": ['--dataset_train_path={}'.format(dataset_train.path.replace('/gcs/','gs://')),
                      '--dataset_validation_path={}'.format(dataset_validation.path.replace('/gcs/','gs://'))]
         }
        #"python_package_spec": {
        #    "executor_image_uri": executor_image_uri,
            #"package_uris": [package_uri],
            #"python_module": python_module,
           # "args": [],
       # },
    }

    # hyperparameter_tuning_job
    hyperparameter_tuning_job = {
        "display_name": display_name,
        "max_trial_count": 4,
        "parallel_trial_count": 2,
        "study_spec": {
            "metrics": [metric],
            "parameters": [parameter_learning_rate,parameter_iterations],
            "algorithm": aiplatform.gapic.StudySpec.Algorithm.RANDOM_SEARCH,
        },
        "trial_job_spec": {"worker_pool_specs": [worker_pool_spec]},
    }
    parent = f"projects/{project}/locations/{location}"
    response = client.create_hyperparameter_tuning_job(parent=parent, hyperparameter_tuning_job=hyperparameter_tuning_job)
    hyperparameter_tuning_job_id = response.name.split('/')[-1]
    
    name = client.hyperparameter_tuning_job_path(
        project=project,
        location=location,
        hyperparameter_tuning_job=hyperparameter_tuning_job_id,
    )
    
    while(1): 
        status_val = client.get_hyperparameter_tuning_job(name=name)
        print(status_val.state.name)
        if (status_val.state.value == 4) | (status_val.state.value == 5) :
            break
        time.sleep(60)
    
    exp_values_list = client.get_hyperparameter_tuning_job(name=name).trials
    exp_dic= {}
    for x in exp_values_list:
        exp_dic[x.id]=x.final_measurement.metrics[0].value
        
    best_exp_id = max(exp_dic, key=exp_dic.get)
   
    best_hp_values.metadata["best_iteration"] = exp_values_list[int(best_exp_id)-1].parameters[0].value
    best_hp_values.metadata["best_learning_rate"] = exp_values_list[int(best_exp_id)-1].parameters[1].value
    