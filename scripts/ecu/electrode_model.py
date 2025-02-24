# Golas: 모듈 경로 추가
import sys
import os

model_upgrading_path = os.path.join("..","..","src")
sys.path.append(model_upgrading_path)

#  moduel
from model import ecu_efficiency_model

# 모델 저장 경로 설정
model_save_path = r'C:\Users\pc021\Desktop\싸인랩 프로젝트\테크로스_ AI 솔루션 개발 실증 지원 사업\자료\모델 관리\성능 추적 관리\모델\ECU\ecu_model_v3.0.0'

# 1. ECU 데이터 준비
ecu_instance = ecu_efficiency_model.EcuModelData()
ecu_instance.initialize_experiment()
training_data = ecu_instance.prepare_training_data()

# 데이터 확인
print("\n[INFO] 데이터 정보:")
training_data.info()

# 2. 모델 학습
print("\n[INFO] 모델 학습 중...")
ecu_instance.train_xgboost_regression_model()

# 3. 모델 저장 
print("\n[INFO] 모델 저장 중...")
ecu_instance.save_model_to_pickle(model_save_path)
print(f"모델이 저장되었습니다: {model_save_path}")