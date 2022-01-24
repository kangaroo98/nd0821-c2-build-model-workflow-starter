'''
Main module containing all MLOps steps to generate a ML model,
which will determine the best price prediction for properties:
- download
- basic cleaning
- data check
- data split
- train random forest
- test regression

Author: Oliver
Date: 2022, Jan

'''
import json

import mlflow
import tempfile
import os
import wandb
import hydra
from omegaconf import DictConfig

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # Steps to execute
    # the steps defined in the config file can be overridden by passing the steps 
    # when calling the module (e.g. python main.py main.steps=download) 
    active_steps = config['main']['steps']
    logger.info(f"Active steps in this ML pipeline: {active_steps}")

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config['etl']['sample'],
                    "artifact_name": config['artifacts']['raw_data_artifact_name'],
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "basic_cleaning"),
                "main",
                parameters={
                    "input_artifact": f"{config['artifacts']['raw_data_artifact_name']}:latest",
                    "output_artifact": config['artifacts']['cleaned_data_artifact_name'],
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
                    "csv": f"{config['artifacts']['cleaned_data_artifact_name']}:latest",
                    "ref": f"{config['artifacts']['cleaned_data_artifact_name']}:reference",
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
                    "input": f"{config['artifacts']['cleaned_data_artifact_name']}:latest",
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

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "src", "train_random_forest"),
                "main",
                parameters={
                    "trainval_artifact": f"{config['artifacts']['trainval_data_artifact_name']}:latest",
                    "val_size": config['modeling']['val_size'],
                    "random_seed": config['modeling']['random_seed'],
                    "stratify_by": config['modeling']['stratify_by'],
                    "rf_config": rf_config,
                    "max_tfidf_features": config['modeling']['max_tfidf_features'],
                    "output_artifact": config['artifacts']['model_artifact_name']
                }
            )

        if "test_regression_model" in active_steps:

            # regression test
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(), "components", "test_regression_model"),
                #f"{config['main']['components_repository']}/test_regression_model",
                "main",
                parameters={
                    "mlflow_model": f"{config['artifacts']['model_artifact_name']}:prod",
                    "test_dataset": f"{config['artifacts']['test_data_artifact_name']}:latest"
                }
            )


if __name__ == "__main__":
    go()
