# basic
import pandas as pd
import pickle

# ml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# vl
import matplotlib.pyplot as plt

# mlflow
import mlflow

# mlflow set
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tc_ai_tro")
mlflow.sklearn.autolog() 

class DataHandler:
    """
    데이터 전처리 및 학습 데이터 생성 클래스.
    """

    def __init__(self, data:pd.DataFrame):
        self.data = data
        self.sensor_name = 'TRO'
        mlflow.xgboost.autolog() 
    
    def initialize_experiment(self):
        mlflow.set_tracking_uri('http://127.0.0.1:5000') # MLflow Tracking 서버 URI 설정
        
        # 실험 존재 여부 확인 및 생성
        if not mlflow.get_experiment_by_name(f"{self.sensor_name}_model"):
            mlflow.create_experiment(f"{self.sensor_name}_model")
        # 설정한 실험 활성화
        mlflow.set_experiment(f"{self.sensor_name}_model")

    @staticmethod
    def preprocessing_label_col(df: pd.DataFrame) -> pd.DataFrame:
        """
        label 컬럼을 이진화 및 정리하는 함수.
        
        Parameters:
            - df (pd.DataFrame): 전처리할 데이터프레임
        
        Returns:
            - pd.DataFrame: 전처리된 데이터프레임
        """
        df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
        df.dropna(inplace=True)

        return df
    

    @staticmethod
    def all_zero(row):
        return row.sum() == 0
    
    
    def create_learning_data(self) -> pd.DataFrame:
        selected_columns = [
            'CSU', 'STS', 'FTS', 'FMU', 'CURRENT', 'TRO_MIN', 'TRO_MEAN', 'TRO_MAX',
            'TRO_DIFF_MIN', 'TRO_DIFF_MEAN', 'TRO_DIFF_MAX', 'TRO_NEG_COUNT',
            'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION',
            'RE_CROSS_CORRELATION_COUNT', 'STEEP_LABEL', 'SLOWLY_LABEL',
            'OUT_OF_WATER_STEEP', 'HUNTING', 'TIME_OFFSET'
        ]

        self.data = self.data[selected_columns]

        self.data['classification'] = self.data[['STEEP_LABEL', 'SLOWLY_LABEL', 'OUT_OF_WATER_STEEP', 'HUNTING', 'TIME_OFFSET']].apply(self.all_zero, axis=1)
        self.data = self.data.set_index(selected_columns[:-5] + ['classification']).stack().reset_index()
        
        self.data.columns = selected_columns[:-5] + ['classification', 'label_name', 'exist']
        self.data = self.data[~((self.data['classification'] == False) & (self.data['exist'] == 0))]
        self.data.drop_duplicates(subset=selected_columns[:-5], inplace=True)

        true_index = self.data[self.data['classification'] == True].index
        self.data.loc[true_index, 'label_name'] = 'Normal'
        
        label_encoding = LabelEncoder()
        self.data['label'] = label_encoding.fit_transform(self.data['label_name'])
    
        self.data = self.data[['CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX',
                               'TRO_NEG_COUNT','PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT','label_name','label']]
        self.data = self.preprocessing_label_col(self.data)

        return self.data
    
    

class ModelHandler:
    """ 모델 학습 및 평가 클래스스
    """
    def __init__(self, model):
        self.model = model

    def train_and_evaluate(self, data: pd.DataFrame) -> None: 
        # mlflow 적용
        with mlflow.start_run():
            X = data.drop(columns=['label']).values
            y = data['label'].values.ravel()

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            X_train = X_train[:, :-1]
            label_name = X_test[:, -1]
            X_test = X_test[:, :-1]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)

            report_dict  = classification_report(y_test, y_pred, output_dict=True)

            mlflow.log_metric('accuracy_on_test', accuracy)
            mlflow.log_metric('precision_on_test', report_dict ['macro avg']['precision'])
            mlflow.log_metric('recall_on_test', report_dict ['macro avg']['recall'])
            mlflow.log_metric('f1-score_on_test', report_dict ['macro avg']['f1-score'])

            conf_matrix = confusion_matrix(y_test, y_pred)
            with open("confusion_matrix.txt", "w") as f:
                f.write(str(conf_matrix))
            mlflow.log_artifact("confusion_matrix.txt")

        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Visualizer().plot_feature_importance(self.model, data.drop(columns='label').columns)

        # X_test = pd.DataFrame(X_test, columns=data.drop(columns=['label_name', 'label']).columns)
        # X_test['Actual'] = y_test
        # X_test['PRED'] = y_pred
        # X_test['label_name'] = label_name


class ModelPersistence:
    """
    모델 저장 및 불러오기 클래스.
    """

    @staticmethod
    def save_model_to_pickle(model, file_path: str) -> None:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"모델이 {file_path}에 성공적으로 저장되었습니다.")


    @staticmethod
    def load_model_from_pickle(file_path: str):
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"모델이 {file_path}에서 성공적으로 불러와졌습니다.")
        return model
    
    
class Visualizer:
    """
    모델 피처 중요도 시각화 클래스.
    """
    @staticmethod
    def plot_feature_importance(model, feature_names):
        importance = model.feature_importances_
        features_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        fig = plt.figure(figsize=(10, 6))
        plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importance')
        plt.gca().invert_yaxis()
        mlflow.log_figure('features_importance.png')
        plt.show()
        plt.close(fig)
    