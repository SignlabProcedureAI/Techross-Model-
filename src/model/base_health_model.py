# basic
import pandas as pd
import pickle
import numpy as np

# model
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# type hinting
from typing  import Tuple
from sklearn.base import BaseEstimator

# vl
import matplotlib.pyplot as plt
import seaborn as sns

# mlflow
import mlflow

class BaseModel:
    def __init__(self):
        mlflow.sklearn.autolog() 

    def initialize_experiment(self):
        mlflow.set_tracking_uri('http://127.0.0.1:5000') # MLflow Tracking 서버 URI 설정
        
        # 실험 존재 여부 확인 및 생성
        if not mlflow.get_experiment_by_name(f"{self.sensor_name}_model"):
            mlflow.create_experiment(f"{self.sensor_name}_model")
        # 설정한 실험 활성화
        mlflow.set_experiment(f"{self.sensor_name}_model")

    def preprocess_sensor_data(self, sensor_name: str) -> pd.DataFrame:
        self.data  = self.data[self.data['HEALTH_SCORE'].notna()]
        self.data = self.data[self.data['HEALTH_SCORE']>0]
        self.data = self.data[(self.data['TRO']>0) & (self.data['TRO']<=8) & (self.data['CURRENT']>0) & (self.data[f'{sensor_name}_MIN']>0)]
    
    def generate_train_data(self, cols: list) -> pd.DataFrame:
        self.slected_df = self.data[cols]   
        return self.slected_df
    
    def train_xgboost_regression_model(self) -> None:
        """
        XGBoost 회귀 모델을 학습하는 함수.
        """
        with mlflow.start_run(): 
            X = self.slected_df.drop('HEALTH_SCORE', axis=1)  
            y = self.slected_df['HEALTH_SCORE']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                
            self.model = GradientBoostingRegressor(
                learning_rate=0.1,
                n_estimators=100,  # 트리 개수
                max_depth=5,  # 트리 최대 깊이
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)

            # 모델 명시적 저장 추가 
            # mlflow.sklearn.log_model(self.model, "xgboost_model")
            # print("✅ 모델이 MLflow에 성공적으로 저장되었습니다.")
            
            self.y_pred = self.model.predict(self.X_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
            r2 = r2_score(self.y_test, self.y_pred)
            print(f'Roots mean Squared Error (RMSE): {rmse}')
            print(f'R^2 Score: {r2}')
            
            mlflow.log_metric('rmse_on_test', rmse)
            mlflow.log_metric('r2_on_test', r2)

            self.plot_regression_results()
            self.plot_feature_importance_xgboost()
            categorize_data = self.convert_continuous_to_f1()
            self.classification_metrics(categorize_data['STATE'], categorize_data['PRED_STATE'])
        
    def save_model_to_pickle(self, file_path: str) -> None:
        """
        모델을 피클 파일로 저장하는 함수.

        Args:
        - model: 저장할 모델 객체 (예: 학습된 모델)
        - file_path: 저장할 피클 파일의 경로
        """
        with open(file_path, 'wb') as file:
            pickle.dump(self.model, file)
        print(f"모델이 {file_path}에 성공적으로 저장되었습니다.")

    def load_model_from_pickle(self, file_path: str) -> BaseEstimator:
        """
        피클 파일에서 모델을 불러오는 함수.

        Args:
        - file_path: 불러올 피클 파일의 경로

        Returns:
        - model: 불러온 모델 객체
        """
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        print(f"모델이 {file_path}에서 성공적으로 불러와졌습니다.")
        return model

    def plot_regression_results(self):
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

        # # 산점도: 실제값 vs 예측값
        # plt.subplot(1, 2, 1)
        # sns.scatterplot(x=self.y_test, y=self.y_pred, alpha=0.5)
        # plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)], color='red', linestyle='--')
        # plt.xlabel('Actual Values')
        # plt.ylabel('Predicted Values')
        # plt.title('Actual vs Predicted Values') 

        # # 오차 분포 히스토그램
        # plt.subplot(1, 2, 2)
        # sns.histplot(self.y_test - self.y_pred, bins=30, kde=True, color='blue')
        # plt.xlabel('Error (Actual - Predicted)')
        # plt.title('Error Distribution')

        plt.tight_layout()
        mlflow.log_figure(fig, "regression_results.png") 
        plt.show()
        plt.close(fig)  # 메모리 해제
        
    def plot_feature_importance_xgboost(self) -> None:
        print("\n[INFO] 모델 결과 시각화...")

        # 피처 중요도 추출
        feature_importances = self.model.feature_importances_
        features = self.X_train.columns  # 피처 이름

        # 데이터프레임으로 정렬하여 피처 중요도를 보기 쉽게 정리
        importance_df = pd.DataFrame({
            'Feature': features,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)

        # 피처 중요도 시각화
        fig = plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Feature Importances from XGBRegressor')
        plt.gca().invert_yaxis()  # 중요도가 높은 피처가 위에 오도록 순서 뒤집기
        mlflow.log_figure(fig, "regression_importances.png") 
        plt.show()
        plt.close(fig)  # 메모리 해제

    def convert_continuous_to_f1(self) -> pd.DataFrame:
        """ 연속형 변수 → f1 스코어 변경
        """
        category_dict = {'CSU': 97, 'STS':60, 'FTS':85, 'FMU':81, 'ECU':67,'ANU':33}

        sensor_limit = category_dict[self.sensor_name]
        metric_data = self.y_test.to_frame()
        metric_data['PRED'] = self.y_pred
        metric_data['STATE'] = 0
        metric_data.loc[metric_data['HEALTH_SCORE']>sensor_limit,'STATE'] = 1
        metric_data['PRED_STATE'] = 0
        metric_data.loc[metric_data['PRED']>sensor_limit,'PRED_STATE'] = 1
        
        return metric_data
            
    def classification_metrics(self, y_true, y_pred):
        """
        정확도, 정밀도, 재현율, F1 스코어를 계산하는 함수.
        
        Args:
        - y_true: list or numpy array, 실제 라벨 값
        - y_pred: list or numpy array, 예측 라벨 값
        
        Returns:
        - metrics: dict, 정확도, 정밀도, 재현율, F1 스코어
        """

        print("\n[INFO] 모델 성능 평가...")

        # 혼동 행렬 구하기
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 정확도 (Accuracy)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # 정밀도 (Precision)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # 재현율 (Recall)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 스코어
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # 결과 출력
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1_score,
            'True Positives (TP)': tp,
            'True Negatives (TN)': tn,
            'False Positives (FP)': fp,
            'False Negatives (FN)': fn
        }
        
        mlflow.log_metric('accuracy_on_test', accuracy)
        mlflow.log_metric('precision_on_test', precision)
        mlflow.log_metric('recall_on_test', recall)
        mlflow.log_metric('f1_on_test', f1_score)

        # 결과 프린트
        print(f"정확도 (Accuracy): {accuracy:.4f}")
        print(f"정밀도 (Precision): {precision:.4f} ({tp} / {tp + fp} 양성 예측)")
        print(f"재현율 (Recall): {recall:.4f} ({tp} / {tp + fn} 실제 양성)")
        print(f"F1 스코어 (F1 Score): {f1_score:.4f}")
        print(f"TP (True Positives): {tp}")
        print(f"TN (True Negatives): {tn}")
        print(f"FP (False Positives): {fp}")
        print(f"FN (False Negatives): {fn}")

        
        
        return metrics

    def plot_histogram(self, bins=10, color='blue', ylabel='Frequency', title='Histogram'):
        """
        Seaborn을 이용해 히스토그램을 그리는 함수

        :param data: 히스토그램으로 그릴 데이터 (리스트나 배열 형태)
        :param bins: 히스토그램의 막대 개수 (기본값: 10)
        :param color: 히스토그램 막대 색상 (기본값: 'blue')
        :param xlabel: X축 라벨 (기본값: 'Value')
        :param ylabel: Y축 라벨 (기본값: 'Frequency')
        :param title: 그래프 제목 (기본값: 'Histogram')
        """
        # 히스토그램 그리기
        sns.histplot(x=self.sensor_name, data=self.slected_df, bins=bins, color=color)

        # 그래프에 라벨과 제목 추가
        plt.xlabel(f'{self.sensor_name}')
        plt.ylabel(ylabel)
        plt.title(title)

        # 그래프 표시
        plt.show()


