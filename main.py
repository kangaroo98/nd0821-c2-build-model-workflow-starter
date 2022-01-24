'''
Main module containing all MLOps steps to generate a ML model,
which will determine the best price prediction for properties:
- download -> components/get_data 
- basic_cleaning -> src/basic_cleaning
- data_check -> src/data_check
- data_split -> components/train_val_split
- train_random_forest -> src/train_random_forest
- test_regression_model -> components/test_regression 
!!!ATTENTION!!! Updated test_regression_model in my git repo to reflect the inference signature
NOTE: test regression is not included in the config.yaml in order not
to run it by mistake. You first need to promote a model export to "prod"
before you can run it, then you need to run this step explicitly.

Author: Oliver
Date: 2022, Jan

'''
import json
import os
import tempfile
import logging
import mlflow
import hydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):
    '''
    main workflow steps, by default it will run:
    - download
    - basic_cleaning
    - data_check
    - data_split
    - train_random_forest

    NOTE: test_regression_model is not included in order not
    to run it by mistake. You first need to promote a model export to "prod"
    before you can run it, then you need to run this step explicitly.

    '''
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    # the steps defined in the config file can be overridden by passing the steps
    # as parameter (e.g. mlflow run . -P steps=download,basic_cleaning)
    active_steps = config['main']['steps']
    logger.info(f"Active steps in this ML pipeline: {active_steps}")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Temporary Directory (rf_config): {tmp_dir}")

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config['etl']['sample'],
                    "artifact_name": config['main']['raw_data_artifact_name'],
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": f"{config['main']['raw_data_artifact_name']}:latest",
                    "output_artifact": config['main']['cleaned_data_artifact_name'],
                    "output_type": "cleaned_data",
                    "output_description": "Data with outliers and null values removed",
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_check" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "data_check"),
                "main",
                parameters={
                    "csv": f"{config['main']['cleaned_data_artifact_name']}:latest",
                    "ref": f"{config['main']['cleaned_data_artifact_name']}:reference",
                    "kl_threshold": config['data_check']['kl_threshold'],
                    "min_price": config['etl']['min_price'],
                    "max_price": config['etl']['max_price']
                },
            )

        if "data_split" in active_steps:
            _ = mlflow.run(
                f"{config['main']['components_repository']}/train_val_test_split",
                "main",
                parameters={
                    "input": f"{config['main']['cleaned_data_artifact_name']}:latest",
                    "test_size": config['modeling']['test_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by']
                },
            )

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the
            # train_random_forest step

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": f"{config['main']['trainval_data_artifact_name']}:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": config['main']['model_artifact_name']
                }
            )

        if "test_regression_model" in active_steps:

            # regression test !!!ATTENTION!!! Update in my git repo to reflect inference signature 
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                #f"{config['main']['components_repository']}/test_regression_model",
                "main",
                parameters={
                    "mlflow_model": f"{config['main']['model_artifact_name']}:prod",
                    "test_dataset": f"{config['main']['test_data_artifact_name']}:latest"
                }
            )


if __name__ == "__main__":
    go()
