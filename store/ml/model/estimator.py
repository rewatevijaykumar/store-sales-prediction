import os
from store.constant.training_pipeline_config import MODEL_FILE_NAME, SAVED_MODEL_DIR

class StoreModel:
    def __init__(self, preprocessor,model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise e
        
class ModelResolver:
    def __init__(self,model_dir=SAVED_MODEL_DIR):
        try:
            self.model_dir = model_dir
        except Exception as e:
            raise e
    
    def get_best_model_path(self)->str:
        try:
            timestamp = list(map(int,os.listdir(self.model_dir)))
            latest_timestamp = max(timestamp)
            latest_model_path = os.path.join(self.model_dir,f'{latest_timestamp}',MODEL_FILE_NAME)
            return latest_model_path
        except Exception as e:
            raise e
        
    def is_model_exists(self)->bool:
        try:
            if not os.path.exists(self.model_dir):
                return False
            
            timestamps = os.listdir(self.model_dir)
            if len(timestamps) == 0:
                return False
            latest_model_path = self.get_best_model_path()
            if not os.path.exists(latest_model_path):
                return False
            return True
        
        except Exception as e:
            raise e    
