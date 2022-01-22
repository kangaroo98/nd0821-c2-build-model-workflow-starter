#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb
import os


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    # Drop outliers
    logger.info("Dropping the outliers with a defined price range")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    # Convert last_review to datetime
    logger.info("Converting column last_review to date format")
    df['last_review'] = pd.to_datetime(df['last_review'])

    # Assuming that in production all names and hostnames are filled properly
    # set it to an empty string
    df['name'].fillna("dummy", inplace=True)
    df['host_name'].fillna("dummy", inplace=True)
    logger.info(
        f"Removed missing values for columns name and host_name (NaN={df['name'].isnull().values.any()})")

    # upload cleand data as an artifact to WB
    logger.info("Storing cleaned data")
    df.to_csv(args.output_artifact)

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

    args = parser.parse_args()

    go(args)
