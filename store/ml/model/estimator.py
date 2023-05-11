import os
from store.constant.training_pipeline_config import MODEL_FILE_NAME

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