"""
XGBoost 및 선택적 LSTM 모델을 위한 모델 학습 모듈입니다.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import joblib
import warnings

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# 선택적 PyTorch 임포트
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("경고: PyTorch를 사용할 수 없습니다. LSTM 학습이 비활성화됩니다.")

# 패키지 실행을 위한 절대 임포트
try:
    from src.data_loader import load_or_download
    from src.features import add_basic_features, make_tabular_dataset, make_supervised_dataset
    from src.config import DATA_DIR, MODEL_DIR, XGB_PARAMS, LSTM_PARAMS, DATA_START, DATA_END, set_global_seed
except ImportError:
    # 로컬 실행을 위한 상대 임포트 (fallback)
    from data_loader import load_or_download
    from features import add_basic_features, make_tabular_dataset, make_supervised_dataset
    from config import DATA_DIR, MODEL_DIR, XGB_PARAMS, LSTM_PARAMS, DATA_START, DATA_END, set_global_seed

warnings.filterwarnings('ignore')


def train_xgb_for_symbol(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    val_split: float = 0.8,
    verbose: bool = True
) -> Tuple[XGBClassifier, MinMaxScaler, list]:
    """
    주어진 심볼에 대해 XGBoost 모델을 학습합니다.
    
    Args:
        symbol: 주식 티커 심볼
        start: 학습 데이터 시작 날짜
        end: 학습 데이터 종료 날짜
        val_split: 학습/검증 분할 비율
        verbose: 진행 상황 출력 여부
    
    Returns:
        학습된 모델, 적합된 스케일러, 피처 컬럼 이름 리스트
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"{symbol}에 대한 XGBoost 모델 학습")
        print(f"{'='*60}\n")
    
    # 데이터 로드
    if verbose:
        print("데이터 로드 중...")
    df = load_or_download(symbol, DATA_DIR, start, end)
    
    # 피처 추가
    if verbose:
        print("피처 엔지니어링 중...")
    df_features = add_basic_features(df)
    
    # 테이블형 데이터셋 생성
    X, y, dates, feature_cols = make_tabular_dataset(df_features)
    
    if verbose:
        print(f"데이터셋 형태: {X.shape}")
        print(f"피처 수: {len(feature_cols)}")
        print(f"날짜 범위: {dates[0].date()} 부터 {dates[-1].date()}")
        print(f"라벨 분포: 상승={np.sum(y)}, 하락={len(y)-np.sum(y)}")
    
    # 학습/검증 분할 (시계열이므로 셔플 없음)
    split_idx = int(len(X) * val_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    if verbose:
        print(f"\n학습 데이터 크기: {len(X_train)}")
        print(f"검증 데이터 크기: {len(X_val)}")
    
    # 피처 스케일링
    if verbose:
        print("\n피처 스케일링 중...")
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # XGBoost 학습
    if verbose:
        print("\nXGBoost 모델 학습 중...")
    
    model = XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
        verbose=False
    )
    
    # 평가
    if verbose:
        print("\n--- 학습 결과 ---")
    
    y_train_pred = model.predict(X_train_scaled)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    train_acc = accuracy_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    y_val_pred = model.predict(X_val_scaled)
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    if verbose:
        print(f"학습 정확도: {train_acc:.4f}")
        print(f"학습 ROC-AUC: {train_auc:.4f}")
        print(f"검증 정확도: {val_acc:.4f}")
        print(f"검증 ROC-AUC: {val_auc:.4f}")
        print("\n검증 분류 보고서:")
        print(classification_report(y_val, y_val_pred, target_names=['하락', '상승']))
    
    # 피처 중요도
    if verbose:
        print("\n상위 10개 중요 피처:")
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(feature_importance.head(10).to_string(index=False))
    
    # 모델, 스케일러, 피처 컬럼 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    model_path = MODEL_DIR / f"{symbol}_xgb_model.pkl"
    scaler_path = MODEL_DIR / f"{symbol}_xgb_scaler.pkl"
    features_path = MODEL_DIR / f"{symbol}_xgb_features.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    
    if verbose:
        print(f"\n모델 저장됨: {model_path}")
        print(f"스케일러 저장됨: {scaler_path}")
        print(f"피처 정보 저장됨: {features_path}")
    
    return model, scaler, feature_cols


# LSTM 모델 정의 (선택 사항)
if TORCH_AVAILABLE:
    class LSTMNet(nn.Module):
        """시계열 분류를 위한 LSTM 네트워크."""
        
        def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
            super(LSTMNet, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
            
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()
        
        def forward(self, x):
            # x 형태: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # 마지막 출력 사용
            last_output = lstm_out[:, -1, :]
            out = self.fc(last_output)
            out = self.sigmoid(out)
            return out


def train_lstm_for_symbol(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    window_size: int = 20,
    val_split: float = 0.8,
    verbose: bool = True
) -> Tuple[Optional[object], Optional[MinMaxScaler], Optional[list]]:
    """
    주어진 심볼에 대해 LSTM 모델을 학습합니다.
    
    Args:
        symbol: 주식 티커 심볼
        start: 학습 데이터 시작 날짜
        end: 학습 데이터 종료 날짜
        window_size: LSTM 시퀀스 길이
        val_split: 학습/검증 분할 비율
        verbose: 진행 상황 출력 여부
    
    Returns:
        학습된 모델, 적합된 스케일러, 피처 컬럼 이름 리스트
    """
    if not TORCH_AVAILABLE:
        print("PyTorch를 사용할 수 없습니다. LSTM 모델을 학습할 수 없습니다.")
        return None, None, None
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"{symbol}에 대한 LSTM 모델 학습")
        print(f"{'='*60}\n")
    
    # 데이터 로드 및 준비
    if verbose:
        print("데이터 로드 중...")
    df = load_or_download(symbol, DATA_DIR, start, end)
    df_features = add_basic_features(df)
    
    # 윈도우 데이터셋 생성
    X, y, dates = make_supervised_dataset(df_features, window_size=window_size)
    
    if verbose:
        print(f"데이터셋 형태: {X.shape}")
        print(f"윈도우 크기: {window_size}")
        print(f"피처 수: {X.shape[2]}")
    
    # 학습/검증 분할
    split_idx = int(len(X) * val_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # 피처 스케일링 (학습 데이터로 fit, 둘 다 transform)
    scaler = MinMaxScaler()
    X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # DataLoader 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=LSTM_PARAMS['batch_size'], shuffle=False)
    
    # 모델 초기화
    input_size = X_train.shape[2]
    model = LSTMNet(
        input_size=input_size,
        hidden_size=LSTM_PARAMS['hidden_size'],
        num_layers=LSTM_PARAMS['num_layers'],
        dropout=LSTM_PARAMS['dropout']
    )
    
    # 손실 함수 및 최적화기
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LSTM_PARAMS['learning_rate'])
    
    # 학습 루프
    if verbose:
        print("\nLSTM 학습 중...")
    
    model.train()
    for epoch in range(LSTM_PARAMS['epochs']):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"에포크 [{epoch+1}/{LSTM_PARAMS['epochs']}], 손실: {avg_loss:.4f}")
    
    # 평가
    model.eval()
    with torch.no_grad():
        y_train_pred_proba = model(X_train_tensor).numpy()
        y_val_pred_proba = model(X_val_tensor).numpy()
        
        y_train_pred = (y_train_pred_proba > 0.5).astype(int).flatten()
        y_val_pred = (y_val_pred_proba > 0.5).astype(int).flatten()
        
        train_acc = accuracy_score(y_train, y_train_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        train_auc = roc_auc_score(y_train, y_train_pred_proba)
        val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    if verbose:
        print(f"\n--- 학습 결과 ---")
        print(f"학습 정확도: {train_acc:.4f}")
        print(f"학습 ROC-AUC: {train_auc:.4f}")
        print(f"검증 정확도: {val_acc:.4f}")
        print(f"검증 ROC-AUC: {val_auc:.4f}")
    
    # 모델 저장
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{symbol}_lstm_model.pth"
    scaler_path = MODEL_DIR / f"{symbol}_lstm_scaler.pkl"
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    if verbose:
        print(f"\n모델 저장됨: {model_path}")
        print(f"스케일러 저장됨: {scaler_path}")
    
    # 피처 컬럼 정보 반환 (일관성을 위해)
    feature_cols = list(range(input_size))
    
    return model, scaler, feature_cols


if __name__ == "__main__":
    # 글로벌 시드 설정
    set_global_seed(42)
    
    # 설정에서 심볼 리스트 가져오기
    try:
        from src.config import DEFAULT_SYMBOLS
    except ImportError:
        from config import DEFAULT_SYMBOLS

    print(f"=== 총 {len(DEFAULT_SYMBOLS)}개 종목에 대한 모델 학습 시작 ===\n")
    print(f"대상 종목: {', '.join(DEFAULT_SYMBOLS)}\n")
    
    for symbol in DEFAULT_SYMBOLS:
        try:
            print(f"\n>> [{symbol}] 학습 진행 중...")
            
            # XGBoost 모델 학습
            train_xgb_for_symbol(
                symbol=symbol,
                start=DATA_START,
                end=DATA_END,
                verbose=True
            )
            
            # PyTorch가 사용 가능한 경우 LSTM 학습 (선택 사항)
            if TORCH_AVAILABLE:
                train_lstm_for_symbol(
                    symbol=symbol,
                    start=DATA_START,
                    end=DATA_END,
                    window_size=20,
                    verbose=True
                )
                
        except Exception as e:
            print(f"!! {symbol} 학습 중 오류 발생: {e}")
            continue

    print(f"\n{'='*60}")
    print("모든 종목에 대한 학습 프로세스 완료")
    print(f"{'='*60}")
