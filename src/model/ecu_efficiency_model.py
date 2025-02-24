# basic
import pandas as pd
import numpy as np
import pickle

# vl
import matplotlib.pyplot as plt
import seaborn as sns

# ml
from xgboost import XGBRegressor,XGBClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# mlflow
import mlflow

# module
from model_dataline.select_dataset import get_dataframe_from_database


class EcuModelData:
    def __init__(self) -> None:
        self.data = get_dataframe_from_database('ecs_test','tc_ai_electrode_group_v1.1.0',all=True)
        self.sensor_name = 'ECU'
        mlflow.xgboost.autolog()

    def initialize_experiment(self):
        mlflow.set_tracking_uri('http://127.0.0.1:5000') # MLflow Tracking 서버 URI 설정
        
        # 실험 존재 여부 확인 및 생성
        if not mlflow.get_experiment_by_name(f"{self.sensor_name}_model"):
            mlflow.create_experiment(f"{self.sensor_name}_model")
        # 설정한 실험 활성화
        mlflow.set_experiment(f"{self.sensor_name}_model")

    def prepare_training_data(self) -> None:
        self.data  = self.data[self.data['ELECTRODE_EFFICIENCY'].notna()]
        self.data = self.data[self.data['CURRENT'] != 0]
        electrod_df = self.data[self.data['ELECTRODE_EFFICIENCY']>=-100]
        electrod_df = electrod_df[(electrod_df['TRO']>0) & (electrod_df['CURRENT']!=0) & (electrod_df['RATE']!=-1)]
        self.electrod_df = electrod_df[['CSU','STS','FTS','FMU','TRO','RATE','VOLTAGE','CURRENT','ELECTRODE_EFFICIENCY']]

        return self.electrod_df

    def train_xgboost_regression_model(self) -> None:
        """
        XGBoost 회귀 모델을 학습하는 함수.
        """
        with mlflow.start_run(): 
        # 입력 특징(X)와 타깃(y) 분리
            self.X = self.electrod_df.drop('ELECTRODE_EFFICIENCY', axis=1)  # 'target_column'을 예측하려는 대상 열로 교체하세요.
            self.y = self.electrod_df['ELECTRODE_EFFICIENCY']

            # 훈련 세트와 테스트 세트로 분할
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
                
            # XGBoost 회귀 모델 초기화
            self.model = XGBRegressor(
                objective='reg:squarederror',  # 회귀를 위한 XGBoost 설정
                n_estimators=100,              # 트리의 개수
                learning_rate=0.1,             # 학습률
                max_depth=5,                   # 트리의 최대 깊이
                random_state=42                # 난수 시드 고정
            )

            # 모델 학습
            self.model.fit(self.X_train, self.y_train)

            # 테스트 데이터로 예측 수행
            self.y_pred = self.model.predict(self.X_test)

            # 성능 평가
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r2 = r2_score(self.y_test, self.y_pred)

            print(f'Roots mean Squared Error (RMSE): {rmse}')
            print(f'R^2 Score: {r2}')
            
            mlflow.log_metric('rmse_on_test', rmse)
            mlflow.log_metric('r2_on_test', r2)
            
            self.plot_regression_results()
            self.plot_feature_importance_xgboost()
            self.convert_continuous_to_f1()
            self.classification_metrics() 
            
    
    def convert_continuous_to_f1(self):
        """ 연속형 변수 → f1 스코어 변경
        """
        self.metric_data = self.y_test.to_frame()
        self.metric_data['ABNORMAL'] = 0
        self.metric_data.loc[self.metric_data['ELECTRODE_EFFICIENCY']<-60, 'ABNORMAL'] =1 

        self.metric_pred_data = pd.DataFrame(self.y_pred, columns=['ELECTRODE_EFFICIENCY'])
        self.metric_pred_data['ABNORMAL'] = 0
        self.metric_pred_data.loc[self.metric_pred_data['ELECTRODE_EFFICIENCY']<-60, 'ABNORMAL'] =1 

    def classification_metrics(self):
        accuracy = accuracy_score(self.metric_data['ABNORMAL'], self.metric_pred_data['ABNORMAL'])
        print(f"Accuracy: {accuracy:.2f}")

        # 상세 평가 지표 출력
        print("\nClassification Report:")
        print(classification_report(self.metric_data['ABNORMAL'], self.metric_pred_data['ABNORMAL']))

        print("\nConfusion Matrix:")
        conf_matrix = confusion_matrix(self.metric_data['ABNORMAL'], self.metric_pred_data['ABNORMAL'])
        print(conf_matrix)
        with open("confusion_matrix.txt", "w") as f:
                f.write(str(conf_matrix))
        mlflow.log_artifact("confusion_matrix.txt")
        mlflow.log_metric('accuracy_on_test', accuracy) 

    def plot_regression_results(self) -> None:
        """
        회귀 모델의 예측 결과를 시각화하는 함수.

        Args:
        - y_test: 실제 타깃 값 (numpy array 또는 pandas Series).
        - y_pred: 모델의 예측 값 (numpy array 또는 pandas Series).
        """

        print("\n[INFO] 모델 결과 시각화...")

        # 그래프 크기 설정
        fig, axes = plt.subplots(1, 2, figsize=(10, 6))

        # 산점도: 실제값 vs 예측값
        sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.5, ax=axes[0])
        axes[0].plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], 
                    color='red', linestyle='--')
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted Values')

        # 오차 분포 히스토그램
        sns.histplot(self.y_test - self.y_pred, bins=30, kde=True, color='blue', ax=axes[1])
        axes[1].set_xlabel('Error (Actual - Predicted)')
        axes[1].set_title('Error Distribution')

        plt.tight_layout()
        mlflow.log_figure(fig, "regression_results.png") 
        plt.show()
        plt.close(fig)  # 메모리 해제

    def plot_feature_importance_xgboost(self):
        # 피처 중요도 추출
        feature_importances = self.model.feature_importances_
        features = self.X.columns  # 피처 이름

        # 데이터프레임으로 정렬하여 피처 중요도를 보기 쉽게 정리
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # 피처 중요도 시각화
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances from XGBRegressor')
        plt.gca().invert_yaxis()  # 중요도가 높은 피처가 위에 오도록 순서 뒤집기
        plt.show()

    def save_model_to_pickle(model, file_path):
        """
        모델을 피클 파일로 저장하는 함수.

        Args:
        - model: 저장할 모델 객체 (예: 학습된 모델)
        - file_path: 저장할 피클 파일의 경로
        """
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"모델이 {file_path}에 성공적으로 저장되었습니다.")
        