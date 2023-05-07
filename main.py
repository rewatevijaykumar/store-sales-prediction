import argparse
from store.config.pipeline.training import StoreConfig
from store.data_access.store_data import StoreData
from store.exception import CustomException
from store.logger import logger
import os,sys
from store.config.mongo_db_connection import MongoDBClient
from store.pipeline.training_pipeline import TrainingPipeline

def start_training(start=False):
    try:
        if not start:
            return None
        print('Training running')
        TrainingPipeline(store_config=StoreConfig()).run_pipeline()
    except Exception as e:
        raise CustomException(e,sys)

def main(training_status):
    try:
        start_training(start=training_status)
    except Exception as e:
        print(e)
        logger.exception(e)

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--t", default=0, type=int, help="If provided true training will be done else not")

        args = parser.parse_args()

        main(training_status=args.t)
    except Exception as e:
        print(e)
        logger.exception(CustomException(e, sys))
