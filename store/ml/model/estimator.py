import os
from store.constant.training_pipeline_config import MODEL_FILE_NAME

class StoreModel:
    def __init__(self, preprocessor, target_preprocessor,model):
        self.preprocessor = preprocessor
        self.target_preprocessor = target_preprocessor
        self.model = model
    
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_transform = self.target_preprocessor.transform(x_transform)
            y_hat = self.model.predict(x_transform,y_transform)
            return y_hat
        except Exception as e:
            raise e