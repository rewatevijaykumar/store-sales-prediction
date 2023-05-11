import os
import sys
import pandas as pd
import yaml
from store.exception import CustomException
from store.logger import logger
import numpy as np
import dill

def read_yaml_file(file_path:str)->dict:
    """
    Reads a YAML file and returns the contents as a dictionary.
    file_path: str
    """
    try:
        with open(file_path, 'r') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
def write_yaml_file(file_path:str, content:str,replace:bool = False) -> None:
    """
    Create yaml file 
    file_path: str
    """
    try:
        logger.info('Entered write_yaml_file method of Main utils')
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as yaml_file:
            yaml.dump(content,yaml_file)
    except Exception as e:
        raise CustomException(e,sys)
    
def read_data(filepath)->pd.DataFrame:
    '''
    Read the data from csv file and return it as a DataFrame
    '''
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_numpy_array_data(file_path:str, array:np.array):
    '''
    Save the numpy array data to file 
    file_path: str location of the file to save
    array: np.array data to save
    '''
    try:
        logger.info('Entered save_numpy_array_data method of Main utils')
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise CustomException(e,sys)
    
def load_numpy_array_data(file_path:str) -> np.array:
    '''
    Loads numpy array data from file
    file_path: str location of the file to load
    return: np.array data loaded
    '''
    try:
        logger.info('Entered load_numpy_array_data method of Main utils')
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_object(file_path:str, obj:object) -> None:
    '''
    Save the preprocessing or machine learning object 
    '''
    try:
        logger.info('Entered save_object method of Main utils')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
        logger.info('Exited save_object method of Main utils')
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path:str) -> object:
    '''
    Load the preprocessing or machine learning object
    '''
    try:
        if not os.path.exists(file_path):
            raise Exception(f'the file: {file_path} does not exist')
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)