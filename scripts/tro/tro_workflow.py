import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_upgrading_path = os.path.join(current_dir,"..","..","src")
sys.path.append(model_upgrading_path)

# module
from model_dataline.select_dataset import get_dataframe_from_database
from model.tro_ml_manager import DataHandler, ModelHandler

# basic
import pandas as pd

#  ml
from xgboost import XGBClassifier

# mlflow set
import mlflow
# mlflow.create_experiment("tc_ai_tro")

class TroPreprocessing:
    """ Tro 전처리 기능
    """

    def __init__(self):
        self.__tro_group = get_dataframe_from_database('ecs_test', 'tc_ai_fault_group_v1.1.0', all=True)


    @property
    def tro_group(self):
        # 원본 데이터 개수 출력 
        print(f"[INFO] 원본 데이터 개수 : {len(self.__tro_group)}")

    
    def preprocess(self):
        """ [행동] Tro 데이터 전처리
        """
        self.__tro_group['SUM'] = (
            self.__tro_group['STEEP_LABEL'] 
            + self.__tro_group['SLOWLY_LABEL'] 
            + self.__tro_group['HUNTING'] 
            + self.__tro_group['OUT_OF_WATER_STEEP'] 
            + self.__tro_group['TIME_OFFSET'] 
        )

        self.__tro_group['STATE'] = 0
        self.__tro_group.loc[self.__tro_group['SUM']>=1, 'STATE'] = 1

        # 모듈 적용
        data_handler = DataHandler(self.__tro_group)
        data_handler.initialize_experiment()
        self.__tro_group = data_handler.create_learning_data()
        print(f"[INFO] 학습 데이터 개수 : {len(self.__tro_group)}")

        return  self.__tro_group


if __name__ =='__main__':
    tro_pipeline = TroPreprocessing() # 객체 생성
    data = tro_pipeline.preprocess() # 학습 데이터 생성
    
    model_handler = ModelHandler(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
    model_handler.train_and_evaluate(data)