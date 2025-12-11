# 📈 AI Algorithmic Trading System (AI 트레이딩 봇)

이 프로젝트는 머신러닝(XGBoost)과 딥러닝(LSTM) 기술을 활용하여 주식 시장의 움직임을 예측하고, 자동으로 매매를 수행하는 **AI 기반 알고리즘 트레이딩 시스템**입니다.

데이터 수집부터 전처리, 피처 엔지니어링, 모델 학습, 백테스팅, 그리고 리스크 관리 및 모의 투자(Paper Trading)까지 퀀트 트레이딩에 필요한 모든 파이프라인을 포함하고 있습니다.

---

## 🚀 주요 기능 (Key Features)

### 1. 📊 데이터 파이프라인 (`data_loader.py`)
- **Yahoo Finance API 연동**: `yfinance`를 통해 전 세계 주식 데이터를 실시간으로 수집합니다.
- **스마트 캐싱**: 중복 다운로드를 방지하기 위해 로컬에 데이터를 캐싱하여 속도를 최적화합니다.
- **안정성 확보**: 네트워크 오류 발생 시 지수 백오프(Exponential Backoff)를 적용한 재시도 로직이 내장되어 있습니다.

### 2. 🛠️ 피처 엔지니어링 (`features.py`)
주가 데이터(OHLCV)를 기반으로 다양한 기술적 지표를 생성하여 모델의 예측력을 높입니다.
- **추세 지표**: 이동평균(MA), MACD, 볼린저 밴드(Bollinger Bands)
- **모멘텀 지표**: RSI, 모멘텀(Momentum)
- **변동성 지표**: ATR, 변동성(Volatility)
- **거래량 지표**: 이동평균 거래량 비율

### 3. 🤖 멀티 모델 학습 (`model_train.py`)
두 가지 강력한 예측 모델을 지원합니다.
- **XGBoost**: 정형 데이터에 강력한 성능을 보이는 그라디언트 부스팅 모델입니다. 피처 중요도 분석을 통해 어떤 지표가 매매에 중요한지 파악할 수 있습니다.
- **LSTM (Long Short-Term Memory)**: 시계열 데이터의 패턴 학습에 특화된 순환 신경망(RNN) 모델입니다. (PyTorch 기반)

### 4. 🧪 정교한 백테스팅 (`backtest.py`)
과거 데이터를 기반으로 전략의 유효성을 검증합니다.
- **벡터화된 백테스팅**: 빠른 속도로 장기간의 데이터를 시뮬레이션합니다.
- **현실적 가정**: 거래 수수료(Fee)와 슬리피지(Slippage)를 반영하여 실제 수익률을 추정합니다.
- **성과 분석**: 수익률, 샤프 지수(Sharpe Ratio), 최대 낙폭(MDD), 승률 등 다양한 지표를 제공합니다.
- **시각화**: 자산 곡선(Equity Curve) 및 월별 수익률 히트맵을 통해 성과를 직관적으로 확인합니다.

### 5. 🛡️ 리스크 관리 (`risk_management.py` & `portfolio.py`)
단순한 수익 추구를 넘어 자산을 보호하기 위한 로직이 포함되어 있습니다.
- **변동성 타겟팅**: 시장 변동성에 따라 포지션 크기를 동적으로 조절합니다.
- **포트폴리오 최적화**: 동일 비중(Equal Weight) 등 다양한 포트폴리오 구성 방식을 지원합니다.
- **손실 제한**: 최대 낙폭(MDD) 제한 로직을 통해 큰 손실을 방지합니다.

### 6. 💸 라이브/모의 트레이딩 (`live_trading.py`)
- **모의 투자 엔진**: 실제 자금을 투입하기 전, 가상의 자금으로 전략을 실시간 테스트할 수 있는 `PaperBroker`를 제공합니다.
- **자동 리밸런싱**: 매일 최신 데이터를 분석하여 목표 포트폴리오 비중대로 매수/매도 주문을 생성합니다.

---

## 📂 프로젝트 구조 (Project Structure)

```
tradeBot/
├── data/                   # 수집된 주가 데이터 저장소 (Git 제외)
├── models/                 # 학습된 모델 및 스케일러 저장소 (Git 제외)
├── src/                    # 소스 코드 디렉토리
│   ├── __init__.py         # 패키지 초기화
│   ├── config.py           # 프로젝트 설정 (경로, 파라미터 등)
│   ├── data_loader.py      # 데이터 수집 및 로드
│   ├── features.py         # 기술적 지표 생성
│   ├── model_train.py      # 모델 학습 (XGBoost, LSTM)
│   ├── model_predict.py    # 예측 및 신호 생성
│   ├── backtest.py         # 백테스팅 엔진
│   ├── live_trading.py     # 라이브/모의 트레이딩 실행
│   ├── portfolio.py        # 포트폴리오 관리
│   └── risk_management.py  # 리스크 지표 계산
├── .gitignore              # Git 제외 파일 설정
├── requirements.txt        # 의존성 패키지 목록
└── README.md               # 프로젝트 설명서
```

---

## 💻 설치 및 실행 방법 (Installation & Usage)

### 1. 환경 설정
Python 3.10 이상이 필요합니다.

```bash
# 저장소 클론
git clone https://github.com/cel0843/tradeBotTest1.git
cd tradeBotTest1

# 가상환경 생성 (권장)
python -m venv .venv

# 가상환경 활성화 (Windows)
.venv\Scripts\activate

# 의존성 패키지 설치
pip install pandas numpy yfinance scikit-learn xgboost matplotlib joblib torch torchvision torchaudio
```

### 2. 모델 학습 (Training)
기본 설정(`src/config.py`)에 따라 모델을 학습합니다. 기본 심볼은 AAPL, MSFT, GOOGL입니다.

```bash
python -m src.model_train
```

### 3. 백테스팅 (Backtesting)
학습된 모델을 기반으로 과거 성과를 분석합니다.

```bash
python -m src.backtest
```

### 4. 라이브/모의 트레이딩 (Live Trading)
현재 시장 데이터를 바탕으로 매매 신호를 생성하고 모의 투자를 수행합니다.

```bash
python -m src.live_trading
```

---

## ⚙️ 설정 변경 (`src/config.py`)

`src/config.py` 파일에서 프로젝트의 모든 주요 파라미터를 수정할 수 있습니다.

- **`DEFAULT_SYMBOLS`**: 트레이딩할 종목 리스트
- **`DATA_START`**: 학습 데이터 시작일
- **`XGB_PARAMS` / `LSTM_PARAMS`**: 모델 하이퍼파라미터
- **`INITIAL_CAPITAL`**: 백테스트/모의투자 초기 자본금
- **`TARGET_VOLATILITY`**: 리스크 관리 목표 변동성

---

## ⚠️ 면책 조항 (Disclaimer)

이 소프트웨어는 교육 및 연구 목적으로 개발되었습니다. 이 시스템이 제공하는 매매 신호는 수익을 보장하지 않으며, 실제 투자에 대한 책임은 전적으로 사용자에게 있습니다. 금융 시장은 예측 불가능한 위험을 내포하고 있으므로 신중하게 사용하시기 바랍니다.
