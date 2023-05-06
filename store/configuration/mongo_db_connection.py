import pymongo
from store.constant.database import DATABASE_NAME
from store.constant.environment.variable_key import MONGODB_URL_KEY
import certifi
ca= certifi.where()
import os
from dotenv import load_dotenv
load_dotenv()

class MongoDBClient:
    client = None
    def __init__(self, database_name=DATABASE_NAME ) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                if 'localhost' in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name

        except Exception as e:
            raise e