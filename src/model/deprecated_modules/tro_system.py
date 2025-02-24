# basic
import pandas as pd
import pickle

# ml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

# visualize
import matplotlib.pyplot as plt


def load_model_from_pickle(file_path):
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


def save_model_to_pickle(model, file_path) -> None:
    """
    모델을 피클 파일로 저장하는 함수.

    Args:
    - model: 저장할 모델 객체 (예: 학습된 모델)
    - file_path: 저장할 피클 파일의 경로

    Returns:
    - 없음
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"모델이 {file_path}에 성공적으로 저장되었습니다.")



def all_zero(row):
    return row.sum() == 0


def create_learning_data(data):
    
    # 변수 선택
    data = data[['CSU', 'STS', 'FTS', 'FMU', 'CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_NEG_COUNT',
                 'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT','STEEP_LABEL', 'SLOWLY_LABEL', 'OUT_OF_WATER_STEEP', 'HUNTING','TIME_OFFSET']]
    
    # 정상 / 오류 분류
    data['classification'] = data[['STEEP_LABEL', 'SLOWLY_LABEL', 'OUT_OF_WATER_STEEP', 'HUNTING','TIME_OFFSET']].apply(all_zero, axis=1)
    
    # 인덱스 설정
    data = data.set_index(['CSU', 'STS', 'FTS', 'FMU', 'CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_NEG_COUNT',
                           'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT','classification'])
    
    # stack 활용 데이터 쌓기
    data = data.stack().to_frame()
    
    # 인덱스 리셋
    data = data.reset_index()
    
    # 컬럼 재 설정
    data.columns =['CSU', 'STS', 'FTS', 'FMU', 'CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_NEG_COUNT',
                   'PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT','classification','label_name','exist']
    
    # 'classification'이 False인 행들 중에서 'exist'가 0인 행들만 필터링
    to_delete = (data['classification'] == False) & (data['exist'] == 0)
    
    # 해당 행들을 삭제
    df_cleaned = data[~to_delete]
    
    # 라벨 이름 설정
    true_index=  df_cleaned[df_cleaned['classification']==True].index
    df_cleaned.loc[true_index,'label_name'] = 'Normal' 
    
    # 삭제 조건: subset 리스트의 컬럼들이 모두 동일한 값을 가질 때 
    subset = ['CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_NEG_COUNT','PEAK_VALLEY_INDICES_SUM', 
              'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT']
    
    # subset에 지정된 컬럼들의 값이 모두 동일한 중복된 행들을 삭제
    df_cleaned = df_cleaned.drop_duplicates(subset=subset)
    
    # 라벨 인코딩
    label_encoding = LabelEncoder() 
    df_cleaned['label'] = label_encoding.fit_transform(df_cleaned['label_name'])
    
    # 컬럼 선택
    fit_data = df_cleaned[['CSU','STS','FTS','FMU','CURRENT','TRO_MIN','TRO_MEAN','TRO_MAX', 'TRO_DIFF_MIN','TRO_DIFF_MEAN','TRO_DIFF_MAX','TRO_NEG_COUNT','PEAK_VALLEY_INDICES_SUM', 'CROSS_CORRELATION', 'RE_CROSS_CORRELATION','RE_CROSS_CORRELATION_COUNT','label_name','label']]
   
    
    return fit_data 



def train_xgboost_simple_model(data):
    """
    불균형 데이터셋에서 XGBoost 모델을 사용하여 불량 탐지를 수행하는 함수입니다.

    Parameters:
    - data: DataFrame, label 컬럼을 포함한 전체 데이터

    Returns:
    - model: 훈련된 XGBoost 모델
    - evaluation: 모델 평가 결과 (dict 형식)
    """

    # 1. 데이터셋 로드
    X = data.drop(columns=['label']).values  # 특징 (features)
    y = data['label'].values.ravel()  # 레이블 (labels)

    # 2. 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = X_train[:,:-1]
    
    label_name = X_test[:,-1]
    X_test = X_test[:,:-1]
    
    # XGBoost 분류기 모델 생성
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 상세 평가 지표 출력
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
#     return model, pd.DataFrame(X_test, columns=data.drop(columns=['label_name', 'label']).columns), y_pred
    
    X_test = pd.DataFrame(X_test,columns = data.drop(columns=['label_name', 'label']).columns)
    X_test['Actual'] = y_test
    X_test['PRED'] = y_pred
    X_test['label_name'] = label_name
    
    return model, X_test

 
def train_random_forest_model(data):
    """
    불균형 데이터셋에서 Random Forest 모델을 사용하여 불량 탐지를 수행하는 함수입니다.

    Parameters:
    - data: DataFrame, label 컬럼을 포함한 전체 데이터

    Returns:
    - model: 훈련된 Random Forest 모델
    - evaluation: 모델 평가 결과 (DataFrame 형식)
    """

    # 1. 데이터셋 로드
    X = data.drop(columns=['label']).values  # 특징 (features)
    y = data['label'].values.ravel()  # 레이블 (labels)

    # 2. 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = X_train[:, :-1]
    
    label_name = X_test[:, -1]
    X_test = X_test[:, :-1]
    
    # Random Forest 분류기 모델 생성
    model = RandomForestClassifier(random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 상세 평가 지표 출력
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    X_test = pd.DataFrame(X_test, columns=data.drop(columns=['label_name', 'label']).columns)
    X_test['Actual'] = y_test
    X_test['PRED'] = y_pred
    X_test['label_name'] = label_name
    
    return model, X_test


def train_extra_trees_model(data):
    """
    불균형 데이터셋에서 Extra Trees 모델을 사용하여 불량 탐지를 수행하는 함수입니다.

    Parameters:
    - data: DataFrame, label 컬럼을 포함한 전체 데이터

    Returns:
    - model: 훈련된 Extra Trees 모델
    - evaluation: 모델 평가 결과 (DataFrame 형식)
    """

    # 1. 데이터셋 로드
    X = data.drop(columns=['label']).values  # 특징 (features)
    y = data['label'].values.ravel()  # 레이블 (labels)

    # 2. 학습 데이터와 테스트 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    X_train = X_train[:, :-1]
    
    label_name = X_test[:, -1]
    X_test = X_test[:, :-1]
    
    # Extra Trees 분류기 모델 생성
    model = ExtraTreesClassifier(random_state=42)

    # 모델 학습
    model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X_test)

    # 모델 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 상세 평가 지표 출력
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    X_test = pd.DataFrame(X_test, columns=data.drop(columns=['label_name', 'label']).columns)
    X_test['Actual'] = y_test
    X_test['PRED'] = y_pred
    X_test['label_name'] = label_name
    
    return model, X_test


def train_xgboost_simple_existing_model(data, model_name):
    """
    불균형 데이터셋에서 XGBoost 모델을 사용하여 불량 탐지를 수행하는 함수입니다.

    Parameters:
    - data: DataFrame, label 컬럼을 포함한 전체 데이터

    Returns:
    - model: 훈련된 XGBoost 모델
    - evaluation: 모델 평가 결과 (dict 형식)
    """

    # 1. 데이터셋 로드
    X = data.drop(columns=['label','label_name']).values  # 특징 (features)
    y = data['label'].values.ravel()  # 레이블 (labels)

#     # 2. 학습 데이터와 테스트 데이터로 분리
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     X_train = X_train[:,:-1]
    
#     label_name = X_test[:,-1]
#     X_test = X_test[:,:-1]
    
    # XGBoost 분류기 모델 생성
  
    model = load_model_from_pickle(rf"C:\Users\pc021\Desktop\dx_project\techross\health_learning_data\health_data\src\update_package\model\{model_name}")
    
#     # 모델 학습
#     model.fit(X_train, y_train)

    # 예측 수행
    y_pred = model.predict(X)

    # 모델 평가
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # 상세 평가 지표 출력
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))
    
    return model, y_pred


# XGBoost 피처 중요도 시각화 함수
def plot_feature_importance(model, feature_names):
    """
    XGBoost 모델의 피처 중요도를 시각화하는 함수.

    Parameters:
    - model: 학습된 XGBoost 모델
    - feature_names: 피처 이름 리스트
    """
    # 피처 중요도 추출
    importance = model.feature_importances_
    
    # 중요도와 피처 이름을 데이터프레임으로 변환
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)
    
    # 중요도 시각화
    plt.figure(figsize=(10, 6))
    plt.barh(features_df['Feature'], features_df['Importance'], color='skyblue')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.title('Feature Importance')
    plt.gca().invert_yaxis()  # 피처 중요도가 높은 순으로 표시
    plt.show()


def preprocessing_label_col(df):
    df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
    
    # 데이터 라벨 역변환
    df['label'] = df['label'].apply(lambda x: 0 if x == 1 else 1)
    
    df.dropna(inplace=True)

    return df

