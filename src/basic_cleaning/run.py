#!/usr/bin/env python
'''
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact

Author: Oliver
Date: 2022, Jan

'''
import os
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    '''
    preprocessing the data
    '''
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df_toclean = pd.read_csv(artifact_path)
    logger.info(f"Read columns: {df_toclean.columns.values}")

    # Drop outliers
    logger.info("Dropping the outliers with a defined price range")
    idx = df_toclean['price'].between(args.min_price, args.max_price) # pylint: disable=unsubscriptable-object
    df_toclean = df_toclean[idx].copy() # pylint: disable=unsubscriptable-object

    # Convert last_review to datetime
    logger.info("Converting column last_review to date format")
    df_toclean['last_review'] = pd.to_datetime(df_toclean['last_review'])

    # Assuming that in production all names and hostnames are filled properly
    # set it to an empty string
    df_toclean['name'].fillna("dummy", inplace=True)
    df_toclean['host_name'].fillna("dummy", inplace=True)
    logger.info(
        f"Removed missing values for columns name and host_name (NaN={df_toclean['name'].isnull().values.any()})")

    # drop outliers of longitude and latitude
    idx = df_toclean['longitude'].between(-74.25, -
                                  73.50) & df_toclean['latitude'].between(40.5, 41.2)
    df_toclean = df_toclean[idx].copy()

    # upload cleand data as an artifact to WB
    logger.info("Storing cleaned data")
    df_toclean.to_csv(args.output_artifact, index=False)
    logger.info(f"Stored Columns: {df_toclean.columns.values}")

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    artifact.wait()

    # clean up
    logger.info("Clean up temp file")
    os.remove(args.output_artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="raw date to be processed",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="cleaned data stored in WB",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="type of the artifact, defaults to cleaned_data",
        required=False
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="cleaned data artifact for further processing",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="min value for data reduction",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="min value for data reduction",
        required=True
    )

    arguments = parser.parse_args()

    go(arguments)
