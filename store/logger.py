import logging
import os
from store.constant import TIMESTAMP

LOG_FILE = f'log-{TIMESTAMP}.log'

logs_path = os.path.join(os.getcwd(), 'logs',LOG_FILE)

os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format='[%(asctime)s] \t%(levelname)s \t%(lineno)d \t%(filename)s \t%(funcName)s() \t%(message)s',
    level=logging.INFO,
)

logger = logging.getLogger("StoreSales")