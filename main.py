from store.exception import CustomException
from store.logger import logging
import os,sys

def main():
    logging.info('starting pipeline')
    try:
        1/0
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__ =='__main__':
    main()