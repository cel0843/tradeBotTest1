# 📈 AI Algorithmic Trading System (AI 트레이딩 봇)

**머신러닝(XGBoost)과 딥러닝(LSTM)을 결합한 고성능 주식 매매 신호 예측 시스템**

이 프로젝트는 과거 주가 데이터를 분석하여 미래의 주가 등락을 예측하고, 최적의 매매 타이밍을 포착하는 AI 봇입니다.  
초보자도 쉽게 따라 할 수 있도록 설계되었으며, 전문가를 위한 확장성도 갖추고 있습니다.

---

## 📚 목차

1. [시작하기 (Installation)](#-시작하기-installation)
2. [사용 방법 (User Guide)](#-사용-방법-user-guide)
    - [1단계: 종목 설정](#1단계-종목-설정)
    - [2단계: AI 모델 학습](#2단계-ai-모델-학습)
    - [3단계: 매매 신호 예측](#3단계-매매-신호-예측)
3. [주요 기능 (Features)](#-주요-기능-features)
4. [프로젝트 구조 (Structure)](#-프로젝트-구조-structure)
5. [주의사항 (Disclaimer)](#-주의사항-disclaimer)

---

## 🚀 시작하기 (Installation)

이 프로젝트를 내 컴퓨터에서 실행하기 위한 준비 과정입니다.

### 1. 필수 프로그램 설치
- **Python 3.8 이상**이 설치되어 있어야 합니다.
- **Git**이 설치되어 있어야 합니다.

### 2. 프로젝트 다운로드 및 설정
터미널(PowerShell 또는 CMD)을 열고 아래 명령어를 순서대로 입력하세요.

```bash
# 1. 프로젝트 복제 (다운로드)
git clone https://github.com/cel0843/tradeBotTest1.git
cd tradeBotTest1

# 2. 가상환경 생성 (권장)
python -m venv .venv

# 3. 가상환경 활성화
# Windows:
.venv\Scripts\activate
# Mac/Linux:
# source .venv/bin/activate

# 4. 필수 라이브러리 설치
pip install -r requirements.txt
```

---

## 📖 사용 방법 (User Guide)

AI를 학습시키고 실제로 매매 신호를 받아보는 방법입니다.

### 1단계: 종목 설정
분석하고 싶은 주식 종목을 설정합니다.
`src/config.py` 파일을 열어 `DEFAULT_SYMBOLS` 리스트를 수정하세요.

```python
# src/config.py 예시

DEFAULT_SYMBOLS = [
    "AAPL", "MSFT", "NVDA",  # 기술주
    "SPY", "QQQ",            # ETF
    "ORCL", "TSLA"           # 추가하고 싶은 종목
]
```
> **Tip**: Yahoo Finance 티커 심볼을 사용합니다. (예: 삼성전자는 '005930.KS')

### 2단계: AI 모델 학습
설정된 모든 종목에 대해 AI가 과거 데이터를 학습합니다.
데이터 다운로드부터 모델 저장까지 자동으로 진행됩니다.

```bash
python -m src.model_train
```

**실행 결과:**
- `models/` 폴더에 각 종목별 모델 파일(`.pkl`, `.pth`)이 저장됩니다.
- 학습 과정에서 정확도(Accuracy)와 ROC-AUC 점수를 확인할 수 있습니다.

### 3단계: 매매 신호 예측
학습된 모델을 사용하여 **오늘의 매매 신호**를 확인합니다.

```bash
python -m src.model_predict
```

**출력 예시:**
```text
[매수 추천 종목 (확률순)]
symbol      close  probability
  NVDA 183.779999     0.699147  <-- 강력 매수 신호
  AAPL 278.779999     0.586623
  MSFT 478.559998     0.564683
```
- **LONG**: 매수 신호 (상승 확률 높음)
- **FLAT**: 관망 신호 (확실하지 않음)
- **Probability**: AI가 예측한 상승 확률 (0.5 이상이면 상승 예측)

---

## 🌟 주요 기능 (Features)

### 1. 앙상블 예측 시스템 (Hybrid AI)
- **XGBoost**: 수치 데이터 분석에 탁월한 머신러닝 모델로, 기술적 지표의 패턴을 분석합니다.
- **LSTM (PyTorch)**: 시계열 데이터(시간의 흐름)를 분석하는 딥러닝 모델로, 차트의 흐름을 읽습니다.
- 두 모델의 장점을 결합하여 예측 정확도를 극대화합니다.

### 2. 자동화된 데이터 파이프라인
- `yfinance`를 통해 전 세계 주식 데이터를 실시간으로 수집합니다.
- 수집된 데이터는 자동으로 전처리(노이즈 제거, 정규화)되어 AI가 학습하기 좋은 형태로 변환됩니다.

### 3. 전문적인 기술적 분석
단순한 가격뿐만 아니라 20여 가지의 보조지표를 활용합니다.
- **추세**: 이동평균선(MA), MACD, 볼린저 밴드
- **모멘텀**: RSI, 스토캐스틱
- **변동성**: ATR, 거래량 변화율

### 4. 리스크 관리 시스템
- **손절매(Stop Loss)** 및 **익절(Take Profit)** 로직 내장
- 변동성이 너무 큰 장세에서는 자동으로 포지션을 축소하는 로직 포함

---

## 📂 프로젝트 구조 (Structure)

```text
tradeBot/
├── models/             # 학습된 AI 모델이 저장되는 곳
├── src/                # 소스 코드 폴더
│   ├── config.py       # 설정 파일 (종목, 기간 등)
│   ├── data_loader.py  # 데이터 수집 모듈
│   ├── features.py     # 기술적 지표 계산
│   ├── model_train.py  # [실행] 모델 학습 스크립트
│   ├── model_predict.py# [실행] 예측 및 신호 생성 스크립트
│   └── ...
├── requirements.txt    # 필요 라이브러리 목록
└── README.md           # 설명서
```

---

## ⚠️ 주의사항 (Disclaimer)

- 본 시스템은 투자를 보조하는 도구이며, **수익을 보장하지 않습니다.**
- 금융 시장은 예측 불가능한 변수가 많으므로, AI의 예측은 참고 자료로만 활용해야 합니다.
- 실제 자금을 운용하기 전에 충분한 모의 투자(Paper Trading)를 거치시길 권장합니다.
- 투자에 대한 모든 책임은 사용자 본인에게 있습니다.
