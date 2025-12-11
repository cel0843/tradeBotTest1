"""
트레이딩 AI 시스템을 위한 설정 모듈입니다.
모든 설정, 경로 및 기본 파라미터를 중앙에서 관리합니다.
"""

from pathlib import Path
from typing import List, Optional
import random
import numpy as np
import os

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# 필요한 디렉토리 생성
MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 기본 트레이딩 심볼 (다양한 섹터 및 ETF 포함)
DEFAULT_SYMBOLS: List[str] = [
    "AAPL", "MSFT", "GOOGL", "NVDA",  # 기술주
    "JPM", "BAC",                     # 금융주
    "JNJ", "PFE",                     # 헬스케어
    "TSLA", "AMZN",                   # 소비재/자동차
    "SPY", "QQQ", "GLD"               # ETF (시장, 기술, 금)
]

# 데이터 다운로드 설정 (기간 확장)
DATA_START: str = "2010-01-01"
DATA_END: Optional[str] = None  # None은 오늘까지를 의미

# 모델 학습 파라미터
TRAIN_VAL_SPLIT: float = 0.8  # 80% 학습, 20% 검증
RANDOM_STATE: int = 42

# XGBoost 파라미터
XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "eval_metric": "logloss"
}

# LSTM 파라미터 (선택 사항)
LSTM_PARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32
}

# 피처 엔지니어링 파라미터
FEATURE_WINDOW_SIZE: int = 20  # 시퀀스 기반 피처를 위한 일수
MA_SHORT: int = 5
MA_LONG: int = 20
RSI_PERIOD: int = 14
VOLUME_PERIOD: int = 5

# 예측 임계값
LONG_THRESHOLD: float = 0.55  # 매수 신호 임계값
FLAT_THRESHOLD: float = 0.45  # 매도/관망 신호 임계값

# 백테스트 파라미터
INITIAL_CAPITAL: float = 100000.0
BACKTEST_FEE_RATE: float = 0.001  # 0.1% 왕복 수수료

# 리스크 관리 파라미터
TARGET_VOLATILITY: float = 0.15  # 15% 연간 목표 변동성
MAX_DRAWDOWN_LIMIT: float = 0.20  # 20% 최대 낙폭 제한
MAX_POSITION_SIZE: float = 0.3  # 포지션 당 최대 30%
RISK_FREE_RATE: float = 0.02  # 샤프 비율 계산용 2% 무위험 이자율

# 포트폴리오 파라미터
MAX_POSITIONS: int = 5  # 최대 동시 보유 포지션 수
REBALANCE_FREQUENCY: str = "daily"  # daily, weekly, monthly

# 라이브 트레이딩 파라미터
PAPER_TRADING: bool = True  # 모의 투자로 시작
INITIAL_PAPER_CASH: float = 100000.0
ORDER_TYPE: str = "market"  # market 또는 limit

# 로깅
LOG_LEVEL: str = "INFO"
LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def set_global_seed(seed: int = 42):
    """
    재현성을 위해 모든 라이브러리의 랜덤 시드를 설정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_model_path(symbol: str, model_type: str = "xgb") -> Path:
    """저장된 모델 파일의 경로를 반환합니다."""
    return MODEL_DIR / f"{symbol}_{model_type}_model.pkl"


def get_scaler_path(symbol: str, model_type: str = "xgb") -> Path:
    """저장된 스케일러 파일의 경로를 반환합니다."""
    return MODEL_DIR / f"{symbol}_{model_type}_scaler.pkl"


def get_feature_cols_path(symbol: str, model_type: str = "xgb") -> Path:
    """저장된 피처 컬럼 정보의 경로를 반환합니다."""
    return MODEL_DIR / f"{symbol}_{model_type}_features.pkl"
