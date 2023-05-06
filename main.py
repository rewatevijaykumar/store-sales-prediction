from store.exception import CustomException
from store.logger import logging
import os,sys
from store.configuration.mongo_db_connection import MongoDBClient

def main():
    logging.info('starting pipeline')
    try:
        mongo = MongoDBClient()
        print(mongo.database)
        print(mongo.database.list_collection_names())
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ =='__main__':
    main()