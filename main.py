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

    # Move to a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:

        if "download" in active_steps:
            # Download file and load in W&B
            _ = mlflow.run(
                f"{config['main']['components_repository']}/get_data",
                "main",
                parameters={
                    "sample": config["etl"]["sample"],
                    "artifact_name": "sample.csv",
                    "artifact_type": "raw_data",
                    "artifact_description": "Raw file as downloaded"
                },
            )

        if "basic_cleaning" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "data_check" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "data_split" in active_steps:
            ##################
            # Implement here #
            ##################
            pass

        if "train_random_forest" in active_steps:

            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            with open(rf_config, "w+") as fp:
                json.dump(dict(config["modeling"]["random_forest"].items()), fp)  # DO NOT TOUCH

            # NOTE: use the rf_config we just created as the rf_config parameter for the train_random_forest
            # step

            ##################
            # Implement here #
            ##################

            pass

        if "test_regression_model" in active_steps:

            ##################
            # Implement here #
            ##################

            pass


if __name__ == "__main__":
    go()
