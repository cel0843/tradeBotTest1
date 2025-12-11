"""
매매 전략 평가를 위한 백테스팅 모듈입니다.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import warnings

# 패키지 실행을 위한 절대 임포트
try:
    from src.risk_management import get_risk_metrics
    from src.data_loader import load_or_download
    from src.model_predict import predict_and_generate_signals
    from src.config import DATA_DIR, DATA_START, DATA_END, INITIAL_CAPITAL, BACKTEST_FEE_RATE, MODEL_DIR
except ImportError:
    # 로컬 실행을 위한 상대 임포트 (fallback)
    from risk_management import get_risk_metrics
    from data_loader import load_or_download
    from model_predict import predict_and_generate_signals
    from config import DATA_DIR, DATA_START, DATA_END, INITIAL_CAPITAL, BACKTEST_FEE_RATE, MODEL_DIR

warnings.filterwarnings('ignore')


def run_backtest(
    prices: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.001,
    slippage: float = 0.0,
    use_close_to_close: bool = True
) -> Dict:
    """
    단일 자산 전략에 대한 벡터화된 백테스트를 실행합니다.
    
    Args:
        prices: 'close' 컬럼이 포함된 DataFrame (최소한)
        signals: 가격과 정렬된 신호 Series (1=매수, 0=관망, -1=매도)
        initial_capital: 초기 자본
        fee_rate: 거래 수수료율 (예: 0.001 = 0.1%)
        slippage: 슬리피지 비율 (각 거래에 대한 추가 비용)
        use_close_to_close: True인 경우 종가 대 종가 수익률 사용
    
    Returns:
        백테스트 결과가 담긴 딕셔너리
    """
    # 가격과 신호 정렬
    df = prices.copy()
    df['signal'] = signals.reindex(df.index, fill_value=0)
    
    # 신호 전방 채우기 (변경될 때까지 포지션 유지)
    df['signal'] = df['signal'].fillna(method='ffill').fillna(0)
    
    # 수익률 계산
    df['close_return'] = df['close'].pct_change()
    
    # 포지션: 전일 신호 사용 (당일 신호로 당일 매매 불가 가정)
    df['position'] = df['signal'].shift(1).fillna(0)
    
    # 포지션 변경 감지 (매매)
    df['position_change'] = df['position'].diff().fillna(df['position'])
    df['trade'] = (df['position_change'] != 0).astype(int)
    
    # 거래 비용 계산
    df['transaction_cost'] = df['trade'] * (fee_rate + slippage) * abs(df['position_change'])
    
    # 전략 수익률 (비용 전)
    df['strategy_return_gross'] = df['position'] * df['close_return']
    
    # 수익률에서 거래 비용 차감
    df['strategy_return_net'] = df['strategy_return_gross'] - df['transaction_cost']
    
    # 자산 곡선 계산
    df['strategy_equity'] = initial_capital * (1 + df['strategy_return_net']).cumprod()
    df['buy_hold_equity'] = initial_capital * (1 + df['close_return'].fillna(0)).cumprod()
    
    strategy_returns = df['strategy_return_net'].dropna()
    buy_hold_returns = df['close_return'].dropna()
    
    # 전략 지표
    strategy_metrics = get_risk_metrics(strategy_returns, df['strategy_equity'].dropna())
    
    # 매수 후 보유(Buy & Hold) 지표
    bh_metrics = get_risk_metrics(buy_hold_returns, df['buy_hold_equity'].dropna())
    
    # 거래 횟수
    n_trades = df['trade'].sum()
    
    # 결과 딕셔너리
    results = {
        'data': df,
        'initial_capital': initial_capital,
        'final_capital': df['strategy_equity'].iloc[-1] if len(df) > 0 else initial_capital,
        'total_return': strategy_metrics['total_return'],
        'annualized_return': strategy_metrics['annualized_return'],
        'annualized_volatility': strategy_metrics['annualized_volatility'],
        'sharpe_ratio': strategy_metrics['sharpe_ratio'],
        'sortino_ratio': strategy_metrics['sortino_ratio'],
        'max_drawdown': strategy_metrics['max_drawdown'],
        'calmar_ratio': strategy_metrics['calmar_ratio'],
        'win_rate': strategy_metrics['win_rate'],
        'n_trades': n_trades,
        'avg_win': strategy_metrics['avg_win'],
        'avg_loss': strategy_metrics['avg_loss'],
        # Buy & Hold 비교
        'bh_total_return': bh_metrics['total_return'],
        'bh_annualized_return': bh_metrics['annualized_return'],
        'bh_sharpe_ratio': bh_metrics['sharpe_ratio'],
        'bh_max_drawdown': bh_metrics['max_drawdown'],
    }
    
    return results


def run_portfolio_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    initial_capital: float = 100000.0,
    fee_rate: float = 0.001
) -> Dict:
    """
    다중 자산 포트폴리오에 대한 백테스트를 실행합니다.
    
    Args:
        prices: 컬럼=심볼, 인덱스=날짜인 DataFrame
        weights: 컬럼=심볼, 인덱스=날짜인 DataFrame (목표 비중)
        initial_capital: 초기 자본
        fee_rate: 거래 수수료율
    
    Returns:
        백테스트 결과가 담긴 딕셔너리
    """
    # 수익률 계산
    returns = prices.pct_change()
    
    # 비중과 수익률 정렬
    common_dates = weights.index.intersection(returns.index)
    weights_aligned = weights.loc[common_dates]
    returns_aligned = returns.loc[common_dates]
    
    # 일별 포트폴리오 수익률 계산
    portfolio_returns = (weights_aligned.shift(1) * returns_aligned).sum(axis=1)
    
    # 회전율 계산 (거래 비용용)
    weight_changes = weights_aligned.diff().abs().sum(axis=1)
    transaction_costs = weight_changes * fee_rate
    
    # 비용 차감 후 순수익률
    portfolio_returns_net = portfolio_returns - transaction_costs
    
    # 자산 곡선
    equity = initial_capital * (1 + portfolio_returns_net).cumprod()
    
    metrics = get_risk_metrics(portfolio_returns_net.dropna(), equity.dropna())
    
    results = {
        'data': pd.DataFrame({
            'portfolio_return': portfolio_returns_net,
            'equity': equity,
            'turnover': weight_changes
        }),
        'initial_capital': initial_capital,
        'final_capital': equity.iloc[-1] if len(equity) > 0 else initial_capital,
        'total_return': metrics['total_return'],
        'annualized_return': metrics['annualized_return'],
        'annualized_volatility': metrics['annualized_volatility'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'avg_turnover': weight_changes.mean(),
        'total_fees': transaction_costs.sum() * initial_capital
    }
    
    return results


def plot_equity_curves(results: Dict, title: str = "전략 성과", figsize: Tuple[int, int] = (12, 6)):
    """
    백테스트 결과로부터 자산 곡선을 그립니다.
    
    Args:
        results: run_backtest의 결과 딕셔너리
        title: 플롯 제목
        figsize: 그림 크기
    """
    df = results['data']
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 자산 곡선
    ax1 = axes[0]
    ax1.plot(df.index, df['strategy_equity'], label='전략', linewidth=2)
    ax1.plot(df.index, df['buy_hold_equity'], label='매수 후 보유', linewidth=2, alpha=0.7)
    ax1.set_ylabel('포트폴리오 가치 ($)')
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 낙폭(Drawdown)
    ax2 = axes[1]
    running_max = df['strategy_equity'].expanding().max()
    drawdown = (df['strategy_equity'] - running_max) / running_max
    ax2.fill_between(df.index, drawdown * 100, 0, alpha=0.3, color='red')
    ax2.set_ylabel('낙폭 (%)')
    ax2.set_xlabel('날짜')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_monthly_returns(results: Dict, figsize: Tuple[int, int] = (12, 6)):
    """
    월별 수익률 히트맵을 그립니다.
    
    Args:
        results: run_backtest의 결과 딕셔너리
        figsize: 그림 크기
    """
    df = results['data']
    
    # 월별로 리샘플링
    monthly_returns = df['strategy_return_net'].resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # 연-월 매트릭스 생성
    monthly_returns_df = monthly_returns.to_frame('return')
    monthly_returns_df['year'] = monthly_returns_df.index.year
    monthly_returns_df['month'] = monthly_returns_df.index.month
    
    pivot = monthly_returns_df.pivot(index='year', columns='month', values='return')
    
    # 히트맵 그리기
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values * 100, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    
    # 눈금 설정
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(['1월', '2월', '3월', '4월', '5월', '6월',
                        '7월', '8월', '9월', '10월', '11월', '12월'])
    ax.set_yticklabels(pivot.index)
    
    # 컬러바 추가
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('수익률 (%)', rotation=270, labelpad=20)
    
    # 값 추가
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            if not np.isnan(pivot.values[i, j]):
                text = ax.text(j, i, f'{pivot.values[i, j]*100:.1f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('월별 수익률 히트맵 (%)')
    plt.tight_layout()
    plt.show()


def print_backtest_summary(results: Dict):
    """
    포맷팅된 백테스트 요약을 출력합니다.
    
    Args:
        results: run_backtest의 결과 딕셔너리
    """
    print("=" * 70)
    print("백테스트 요약")
    print("=" * 70)
    
    print(f"\n--- 자본 ---")
    print(f"초기 자본:              ${results['initial_capital']:>15,.2f}")
    print(f"최종 자본:              ${results['final_capital']:>15,.2f}")
    print(f"총 수익률:              {results['total_return']:>15.2%}")
    
    print(f"\n--- 성과 지표 ---")
    print(f"연환산 수익률:          {results['annualized_return']:>15.2%}")
    print(f"연환산 변동성:          {results['annualized_volatility']:>15.2%}")
    print(f"샤프 비율:              {results['sharpe_ratio']:>15.2f}")
    print(f"소르티노 비율:          {results.get('sortino_ratio', 0):>15.2f}")
    print(f"칼마 비율:              {results.get('calmar_ratio', 0):>15.2f}")
    print(f"최대 낙폭(MDD):         {results['max_drawdown']:>15.2%}")
    
    print(f"\n--- 매매 통계 ---")
    print(f"총 거래 횟수:           {results['n_trades']:>15.0f}")
    print(f"승률:                   {results.get('win_rate', 0):>15.2%}")
    print(f"평균 수익:              {results.get('avg_win', 0):>15.2%}")
    print(f"평균 손실:              {results.get('avg_loss', 0):>15.2%}")
    
    if 'bh_total_return' in results:
        print(f"\n--- 매수 후 보유(B&H) 비교 ---")
        print(f"B&H 총 수익률:          {results['bh_total_return']:>15.2%}")
        print(f"B&H 연환산 수익률:      {results['bh_annualized_return']:>15.2%}")
        print(f"B&H 샤프 비율:          {results['bh_sharpe_ratio']:>15.2f}")
        print(f"B&H 최대 낙폭:          {results['bh_max_drawdown']:>15.2%}")
        
        excess_return = results['total_return'] - results['bh_total_return']
        print(f"초과 수익률:            {excess_return:>15.2%}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("=== 백테스트 테스트 ===\n")
    
    symbol = "AAPL"
    
    # 모델 존재 여부 확인
    model_path = MODEL_DIR / f"{symbol}_xgb_model.pkl"
    
    if not model_path.exists():
        print(f"{symbol}에 대한 모델을 찾을 수 없습니다. 더미 신호로 백테스트를 실행합니다...")
        
        # 데이터 로드
        df = load_or_download(symbol, DATA_DIR, DATA_START, DATA_END)
        
        # 더미 신호 생성 (단순 이동평균 교차)
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        signals = (df['ma5'] > df['ma20']).astype(int)
        
    else:
        print(f"{symbol}에 대해 학습된 모델 사용...")
        
        # 데이터 로드
        df = load_or_download(symbol, DATA_DIR, DATA_START, DATA_END)
        
        # 모델로부터 신호 생성
        predictions = predict_and_generate_signals(symbol, df)
        signals = predictions['signal']
    
    # 백테스트 실행
    print("\n백테스트 실행 중...")
    results = run_backtest(
        prices=df,
        signals=signals,
        initial_capital=INITIAL_CAPITAL,
        fee_rate=BACKTEST_FEE_RATE
    )
    
    # 요약 출력
    print_backtest_summary(results)
    
    # 플롯 생성
    print("\n플롯 생성 중...")
    plot_equity_curves(results, title=f"{symbol} 전략 성과")
    
    try:
        plot_monthly_returns(results)
    except Exception as e:
        print(f"월별 수익률 플롯 생성 실패: {e}")
