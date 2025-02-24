# module
from model_dataline.select_dataset import get_dataframe_from_database
from model.base_health_model import BaseModel


class CsuModelData(BaseModel):
    def __init__ (self) -> None:
        super().__init__()
        self.sensor_name = 'CSU'
        self.data = get_dataframe_from_database('ecs_test','tc_ai_csu_system_health_group_v1.1.0',True)
        self.preprocessed_df = self.preprocess_sensor_data(self.sensor_name)

    def select_model_variables(self):
        variable = [
            'CSU_MIN', 'CSU_MEAN','CSU_MAX',
            'STS','FTS','FMU','TRO','CURRENT',
            'DIFF_MIN','DIFF_MEAN','DIFF_MAX',
            'TREND_SCORE','HEALTH_SCORE'
            ]
        self.preprocessed_df = self.preprocessed_df[variable]
    

class StsModelData(BaseModel):
    def __init__ (self) -> None:
        self.sensor_name = 'STS'
        self.data = get_dataframe_from_database('ecs_test', 'tc_ai_sts_system_health_group_v1.1.0', True)
        self.preprocessed_df = self.preprocess_sensor_data(self.sensor_name)
        
    def select_model_variables(self):
            variable = [
                'STS_MIN', 'STS_MEAN','STS_MAX',
                'CSU','FTS','FMU','TRO','CURRENT',
                'DIFF_MIN','DIFF_MEAN','DIFF_MAX',
                'TREND_SCORE','HEALTH_SCORE'
                ]
            self.preprocessed_df = self.preprocessed_df[variable]


class FtsModelData(BaseModel):
    def __init__ (self) -> None:
        self.sensor_name = 'FTS'
        self.data = get_dataframe_from_database('ecs_test','tc_ai_fts_system_health_group_v1.1.0',True)
        self.preprocessed_df = self.preprocess_sensor_data(self.sensor_name)
    
    def select_model_variables(self):
        variable = [
            'FTS_MIN', 'FTS_MEAN','FTS_MAX',
            'CSU','STS','FMU','TRO','CURRENT',
            'DIFF_MIN','DIFF_MEAN','DIFF_MAX',
            'TREND_SCORE','HEALTH_SCORE'
            ]
        self.preprocessed_df = self.preprocessed_df[variable]


class FmuModelData(BaseModel):
    def __init__ (self):
        self.sensor_name = 'FMU'
        self.data = get_dataframe_from_database('ecs_test','tc_ai_fmu_system_health_group_v1.1.0',True)
        self.preprocessed_df = self.preprocess_sensor_data(self.sensor_name)
        
    def select_model_variables(self):
        variable = [
            'FMU_MIN', 'FMU_MEAN','FMU_MAX',
            'CSU','STS','FTS','TRO','CURRENT',
            'STANDARDIZE_FMU_MIN','STANDARDIZE_FMU_MEAN',
            'STANDARDIZE_FMU_MAX','HEALTH_SCORE'
            ]
        self.preprocessed_df = self.preprocessed_df[variable]