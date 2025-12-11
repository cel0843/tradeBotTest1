"""
다중 자산 포지션 사이징 및 최적화를 위한 포트폴리오 관리 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def equal_weight_portfolio(
    signals: pd.DataFrame,
    max_positions: int = 5,
    min_signal: int = 1
) -> pd.DataFrame:
    """
    신호로부터 동일 비중 포트폴리오를 생성합니다.
    
    Args:
        signals: 인덱스=날짜, 컬럼=심볼, 값=신호(0, 1, -1)인 DataFrame
        max_positions: 보유할 최대 포지션 수
        min_signal: 고려할 최소 신호 값 (기본값 1 = 매수만)
    
    Returns:
        인덱스=날짜, 컬럼=심볼, 값=비중인 DataFrame
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    
    for date in signals.index:
        # 해당 날짜의 신호 가져오기
        date_signals = signals.loc[date]
        
        # 최소 신호 조건 필터링
        eligible = date_signals[date_signals >= min_signal]
        
        # 상위 포지션 선택 (필요 시 신호 강도로 순위 매김 가능)
        n_positions = min(len(eligible), max_positions)
        
        if n_positions > 0:
            # 선택된 포지션 간 동일 비중
            weight = 1.0 / n_positions
            top_symbols = eligible.nlargest(n_positions).index
            weights.loc[date, top_symbols] = weight
    
    return weights


def market_cap_weight_portfolio(
    signals: pd.DataFrame,
    market_caps: pd.Series,
    max_positions: int = 5,
    min_signal: int = 1
) -> pd.DataFrame:
    """
    신호로부터 시가총액 가중 포트폴리오를 생성합니다.
    
    Args:
        signals: 인덱스=날짜, 컬럼=심볼, 값=신호인 DataFrame
        market_caps: 인덱스=심볼, 값=시가총액인 Series
        max_positions: 최대 포지션 수
        min_signal: 고려할 최소 신호 값
    
    Returns:
        포트폴리오 비중이 담긴 DataFrame
    """
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    
    for date in signals.index:
        date_signals = signals.loc[date]
        eligible = date_signals[date_signals >= min_signal]
        
        if len(eligible) > 0:
            # 상위 포지션 선택
            top_symbols = eligible.nlargest(max_positions).index
            
            # 선택된 심볼의 시가총액 가져오기
            caps = market_caps.loc[top_symbols]
            
            # 시가총액으로 가중치 부여
            total_cap = caps.sum()
            if total_cap > 0:
                weights.loc[date, top_symbols] = caps / total_cap
    
    return weights


def risk_parity_weights(
    returns: pd.DataFrame,
    lookback: int = 60
) -> pd.Series:
    """
    역변동성에 기반하여 리스크 패리티 비중을 계산합니다.
    
    Args:
        returns: 컬럼=심볼인 수익률 DataFrame
        lookback: 변동성 계산을 위한 기간
    
    Returns:
        비중 Series (인덱스=심볼)
    """
    # 변동성 계산
    vols = returns.tail(lookback).std()
    
    # 역변동성
    inv_vols = 1.0 / vols
    
    # 합이 1이 되도록 정규화
    weights = inv_vols / inv_vols.sum()
    
    return weights


def mean_variance_optimize(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    target_return: Optional[float] = None,
    risk_aversion: float = 1.0
) -> pd.Series:
    """
    평균-분산 최적화 (향후 구현을 위한 플레이스홀더).
    
    이것은 단순화된 버전입니다. 실제 운영 시에는 PyPortfolioOpt와 같은 라이브러리를 사용하세요.
    
    Args:
        expected_returns: 기대 수익률 Series (인덱스=심볼)
        cov_matrix: 수익률 공분산 행렬
        target_return: 목표 수익률 수준
        risk_aversion: 위험 회피 파라미터
    
    Returns:
        최적 비중 Series
    """
    # 플레이스홀더: 단순 역분산 가중
    # 실제 운영 시에는 적절한 2차 최적화 구현 필요
    
    variances = np.diag(cov_matrix)
    inv_var = 1.0 / variances
    weights = inv_var / inv_var.sum()
    
    return pd.Series(weights, index=expected_returns.index)


def apply_position_limits(
    weights: pd.DataFrame,
    max_weight: float = 0.3,
    min_weight: float = 0.0
) -> pd.DataFrame:
    """
    포트폴리오 비중에 포지션 크기 제한을 적용합니다.
    
    Args:
        weights: 포트폴리오 비중 DataFrame
        max_weight: 포지션 당 최대 비중
        min_weight: 포지션 당 최소 비중
    
    Returns:
        제한이 적용된 비중 DataFrame (재정규화됨)
    """
    limited = weights.copy()
    
    for date in weights.index:
        date_weights = weights.loc[date]
        
        # 제한 적용
        date_weights = date_weights.clip(lower=min_weight, upper=max_weight)
        
        # 합이 1을 초과하면 재정규화
        weight_sum = date_weights.sum()
        if weight_sum > 1:
            date_weights = date_weights / weight_sum
        
        limited.loc[date] = date_weights
    
    return limited


def calculate_portfolio_returns(
    weights: pd.DataFrame,
    returns: pd.DataFrame
) -> pd.Series:
    """
    비중과 자산 수익률로부터 포트폴리오 수익률을 계산합니다.
    
    Args:
        weights: 포트폴리오 비중 DataFrame (인덱스=날짜, 컬럼=심볼)
        returns: 자산 수익률 DataFrame (인덱스=날짜, 컬럼=심볼)
    
    Returns:
        포트폴리오 수익률 Series (인덱스=날짜)
    """
    # 날짜 정렬
    common_dates = weights.index.intersection(returns.index)
    weights_aligned = weights.loc[common_dates]
    returns_aligned = returns.loc[common_dates]
    
    # 가중 수익률 계산
    portfolio_returns = (weights_aligned * returns_aligned).sum(axis=1)
    
    return portfolio_returns


def rebalance_portfolio(
    current_weights: pd.Series,
    target_weights: pd.Series,
    threshold: float = 0.05
) -> Tuple[pd.Series, pd.Series]:
    """
    포트폴리오 리밸런싱에 필요한 매매를 결정합니다.
    
    Args:
        current_weights: 현재 비중 Series (인덱스=심볼)
        target_weights: 목표 비중 Series (인덱스=심볼)
        threshold: 리밸런싱을 트리거할 최소 비중 차이
    
    Returns:
        (trades, new_weights) 튜플, trades는 필요한 변경량
    """
    # 심볼 정렬
    all_symbols = current_weights.index.union(target_weights.index)
    current = current_weights.reindex(all_symbols, fill_value=0.0)
    target = target_weights.reindex(all_symbols, fill_value=0.0)
    
    # 차이 계산
    diffs = target - current
    
    # 차이가 임계값을 초과하는 경우에만 매매
    trades = diffs.copy()
    trades[abs(diffs) < threshold] = 0.0
    
    # 매매 후 새로운 비중
    new_weights = current + trades
    
    return trades, new_weights


def construct_multi_strategy_portfolio(
    strategy_returns: Dict[str, pd.Series],
    allocation: Dict[str, float] = None
) -> pd.Series:
    """
    여러 전략을 단일 포트폴리오로 결합합니다.
    
    Args:
        strategy_returns: 전략 이름 -> 수익률 Series 매핑 딕셔너리
        allocation: 전략 이름 -> 비중 매핑 딕셔너리. None이면 동일 비중.
    
    Returns:
        결합된 포트폴리오 수익률 Series
    """
    if allocation is None:
        # 동일 비중
        n_strategies = len(strategy_returns)
        allocation = {name: 1.0 / n_strategies for name in strategy_returns.keys()}
    
    # 모든 시리즈를 공통 날짜로 정렬
    returns_df = pd.DataFrame(strategy_returns)
    
    # 가중 수익률 계산
    portfolio_returns = sum(
        returns_df[name] * weight
        for name, weight in allocation.items()
    )
    
    return portfolio_returns


def get_top_n_by_signal(
    signals: pd.Series,
    probabilities: pd.Series,
    n: int = 5
) -> List[str]:
    """
    신호 강도와 확률에 따라 상위 N개 심볼을 가져옵니다.
    
    Args:
        signals: 신호 Series (인덱스=심볼)
        probabilities: 확률 Series (인덱스=심볼)
        n: 선택할 심볼 수
    
    Returns:
        상위 N개 심볼 이름 리스트
    """
    # 양수 신호 필터링
    positive = signals[signals > 0]
    
    if len(positive) == 0:
        return []
    
    # 해당 확률 가져오기
    probs = probabilities.loc[positive.index]
    
    # 확률순 정렬 (내림차순)
    top = probs.nlargest(n)
    
    return top.index.tolist()


class Portfolio:
    """
    포지션과 성과를 추적하기 위한 포트폴리오 객체입니다.
    """
    
    def __init__(self, initial_cash: float = 100000.0, symbols: List[str] = None):
        """
        포트폴리오를 초기화합니다.
        
        Args:
            initial_cash: 초기 현금 금액
            symbols: 추적할 심볼 리스트
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {심볼: 수량}
        self.symbols = symbols or []
        self.history = []  # 포트폴리오 스냅샷 리스트
    
    def get_position(self, symbol: str) -> float:
        """심볼에 대한 현재 포지션 크기를 가져옵니다."""
        return self.positions.get(symbol, 0.0)
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """
        총 포트폴리오 가치를 계산합니다.
        
        Args:
            prices: 심볼 -> 현재가 매핑 딕셔너리
        
        Returns:
            총 포트폴리오 가치
        """
        position_value = sum(
            qty * prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_weights(self, prices: Dict[str, float]) -> Dict[str, float]:
        """
        현재 포트폴리오 비중을 가져옵니다.
        
        Args:
            prices: 심볼 -> 현재가 매핑 딕셔너리
        
        Returns:
            심볼 -> 비중 매핑 딕셔너리
        """
        total_value = self.get_total_value(prices)
        
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, qty in self.positions.items():
            position_value = qty * prices.get(symbol, 0.0)
            weights[symbol] = position_value / total_value
        
        return weights
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """
        심볼에 대한 포지션을 업데이트합니다.
        
        Args:
            symbol: 매매할 심볼
            quantity: 수량 변경 (양수=매수, 음수=매도)
            price: 체결 가격
        """
        # 현금 업데이트
        cost = quantity * price
        self.cash -= cost
        
        # 포지션 업데이트
        current_qty = self.positions.get(symbol, 0.0)
        new_qty = current_qty + quantity
        
        if new_qty == 0:
            # 포지션 청산
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = new_qty
    
    def snapshot(self, date, prices: Dict[str, float]) -> Dict:
        """
        현재 포트폴리오 상태의 스냅샷을 찍습니다.
        
        Args:
            date: 현재 날짜
            prices: 현재 가격
        
        Returns:
            포트폴리오 상태가 담긴 딕셔너리
        """
        snapshot = {
            'date': date,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'total_value': self.get_total_value(prices),
            'weights': self.get_weights(prices)
        }
        
        self.history.append(snapshot)
        return snapshot
    
    def get_history_df(self) -> pd.DataFrame:
        """
        포트폴리오 기록을 DataFrame으로 가져옵니다.
        
        Returns:
            포트폴리오 기록이 담긴 DataFrame
        """
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {
                'date': h['date'],
                'cash': h['cash'],
                'total_value': h['total_value']
            }
            for h in self.history
        ])
        
        df.set_index('date', inplace=True)
        return df


if __name__ == "__main__":
    print("=== 포트폴리오 관리 테스트 ===\n")
    
    # 샘플 신호 생성
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    np.random.seed(42)
    signals = pd.DataFrame(
        np.random.choice([0, 1], size=(len(dates), len(symbols))),
        index=dates,
        columns=symbols
    )
    
    print("샘플 신호:")
    print(signals)
    
    # 동일 비중 포트폴리오 테스트
    print("\n--- 동일 비중 포트폴리오 ---")
    weights = equal_weight_portfolio(signals, max_positions=3)
    print(weights)
    
    # 포트폴리오 클래스 테스트
    print("\n--- 포트폴리오 클래스 테스트 ---")
    portfolio = Portfolio(initial_cash=100000.0, symbols=symbols)
    
    prices = {'AAPL': 150.0, 'MSFT': 300.0, 'GOOGL': 140.0}
    
    # 주식 매수
    portfolio.update_position('AAPL', 100, 150.0)
    portfolio.update_position('MSFT', 50, 300.0)
    
    print(f"현금: ${portfolio.cash:,.2f}")
    print(f"포지션: {portfolio.positions}")
    print(f"총 가치: ${portfolio.get_total_value(prices):,.2f}")
    print(f"비중: {portfolio.get_weights(prices)}")
