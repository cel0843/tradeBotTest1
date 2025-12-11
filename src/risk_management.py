"""
포트폴리오 리스크 지표 및 포지션 사이징을 위한 리스크 관리 모듈입니다.
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings

warnings.filterwarnings('ignore')


def calc_annualized_return(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    일일 수익률로부터 연환산 수익률을 계산합니다.
    
    Args:
        daily_returns: 일일 수익률 시리즈
        periods_per_year: 연간 거래 기간 수 (일일 데이터의 경우 252)
    
    Returns:
        연환산 수익률 (소수점, 예: 0.15 = 15%)
    """
    if len(daily_returns) == 0:
        return 0.0
    
    total_return = (1 + daily_returns).prod() - 1
    n_periods = len(daily_returns)
    
    if n_periods == 0:
        return 0.0
    
    annualized = (1 + total_return) ** (periods_per_year / n_periods) - 1
    return float(annualized)


def calc_annualized_volatility(daily_returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    일일 수익률로부터 연환산 변동성을 계산합니다.
    
    Args:
        daily_returns: 일일 수익률 시리즈
        periods_per_year: 연간 거래 기간 수 (일일 데이터의 경우 252)
    
    Returns:
        연환산 변동성 (소수점)
    """
    if len(daily_returns) < 2:
        return 0.0
    
    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(periods_per_year)
    
    return float(annualized_vol)


def calc_sharpe_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    샤프 비율을 계산합니다.
    
    Args:
        daily_returns: 일일 수익률 시리즈
        risk_free_rate: 연간 무위험 이자율 (예: 0.02 = 2%)
        periods_per_year: 연간 거래 기간 수
    
    Returns:
        샤프 비율
    """
    if len(daily_returns) < 2:
        return 0.0
    
    ann_return = calc_annualized_return(daily_returns, periods_per_year)
    ann_vol = calc_annualized_volatility(daily_returns, periods_per_year)
    
    if ann_vol == 0:
        return 0.0
    
    sharpe = (ann_return - risk_free_rate) / ann_vol
    return float(sharpe)


def calc_max_drawdown(equity_curve: pd.Series) -> float:
    """
    자산 곡선으로부터 최대 낙폭(MDD)을 계산합니다.
    
    Args:
        equity_curve: 시간 경과에 따른 포트폴리오 가치 시리즈
    
    Returns:
        최대 낙폭 (양수 소수점, 예: 0.20 = 20% 낙폭)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    # 누적 최대값 계산
    running_max = equity_curve.expanding().max()
    
    # 각 시점에서의 낙폭 계산
    drawdown = (equity_curve - running_max) / running_max
    
    # 최대 낙폭 (가장 작은 음수 값을 양수로 반환)
    max_dd = abs(drawdown.min())
    
    return float(max_dd)


def calc_sortino_ratio(
    daily_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    소르티노 비율을 계산합니다 (샤프와 유사하지만 하방 편차만 사용).
    
    Args:
        daily_returns: 일일 수익률 시리즈
        risk_free_rate: 연간 무위험 이자율
        periods_per_year: 연간 거래 기간 수
    
    Returns:
        소르티노 비율
    """
    if len(daily_returns) < 2:
        return 0.0
    
    ann_return = calc_annualized_return(daily_returns, periods_per_year)
    
    # 하방 편차 (음수 수익률만 고려)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if ann_return > risk_free_rate else 0.0
    
    downside_std = negative_returns.std()
    downside_vol = downside_std * np.sqrt(periods_per_year)
    
    if downside_vol == 0:
        return 0.0
    
    sortino = (ann_return - risk_free_rate) / downside_vol
    return float(sortino)


def calc_calmar_ratio(daily_returns: pd.Series, equity_curve: pd.Series) -> float:
    """
    칼마 비율을 계산합니다 (연환산 수익률 / 최대 낙폭).
    
    Args:
        daily_returns: 일일 수익률 시리즈
        equity_curve: 포트폴리오 가치 시리즈
    
    Returns:
        칼마 비율
    """
    ann_return = calc_annualized_return(daily_returns)
    max_dd = calc_max_drawdown(equity_curve)
    
    if max_dd == 0:
        return 0.0
    
    calmar = ann_return / max_dd
    return float(calmar)


def volatility_target_position_size(
    target_vol: float,
    realized_vol: float,
    base_notional: float = 1.0,
    max_leverage: float = 3.0
) -> float:
    """
    변동성 타겟팅에 기반하여 포지션 크기를 계산합니다.
    
    Args:
        target_vol: 목표 연간 변동성 (예: 0.15 = 15%)
        realized_vol: 최근 실현 변동성 (연환산)
        base_notional: 기본 명목 금액 (기본값 1.0 = 100%)
        max_leverage: 최대 허용 레버리지
    
    Returns:
        포지션 크기 배율 (예: 1.5 = 기본의 150%)
    """
    if realized_vol == 0:
        return base_notional
    
    # 변동성에 반비례하여 포지션 조절
    position_size = base_notional * (target_vol / realized_vol)
    
    # 최대 레버리지로 제한
    position_size = min(position_size, base_notional * max_leverage)
    
    # 최소 임계값 이하로 내려가지 않도록 설정
    position_size = max(position_size, base_notional * 0.1)
    
    return float(position_size)


def apply_drawdown_limit(
    equity_curve: pd.Series,
    max_dd_limit: float = 0.20,
    recovery_threshold: float = 0.95
) -> pd.Series:
    """
    낙폭 제한 적용: 최대 낙폭 초과 시 거래 중단.
    
    Args:
        equity_curve: 포트폴리오 가치 시리즈
        max_dd_limit: 최대 허용 낙폭 (예: 0.20 = 20%)
        recovery_threshold: 거래 재개를 위한 회복 비율 (예: 0.95 = 고점의 95%)
    
    Returns:
        제한이 적용된 자산 곡선 (제한 초과 시 플랫)
    """
    equity_limited = equity_curve.copy()
    running_max = equity_curve.iloc[0]
    stopped = False
    stop_value = None
    
    for i in range(len(equity_curve)):
        current_value = equity_curve.iloc[i]
        
        if not stopped:
            # 누적 최대값 업데이트
            if current_value > running_max:
                running_max = current_value
            
            # 낙폭 확인
            current_dd = (running_max - current_value) / running_max
            
            if current_dd > max_dd_limit:
                # 거래 중단
                stopped = True
                stop_value = current_value
                equity_limited.iloc[i] = stop_value
        else:
            # 거래 재개를 위해 충분히 회복되었는지 확인
            if current_value >= running_max * recovery_threshold:
                # 거래 재개
                stopped = False
                running_max = current_value
                equity_limited.iloc[i] = current_value
            else:
                # 플랫 유지
                equity_limited.iloc[i] = stop_value
    
    return equity_limited


def calculate_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    VaR(Value at Risk)를 계산합니다.
    
    Args:
        returns: 수익률 시리즈
        confidence_level: 신뢰 수준 (예: 0.95 = 95%)
        method: 'historical' 또는 'parametric'
    
    Returns:
        VaR (양수, 예: 0.02 = 2% 잠재 손실)
    """
    if len(returns) == 0:
        return 0.0
    
    if method == 'historical':
        # 역사적 VaR: 경험적 분위수 사용
        var = abs(returns.quantile(1 - confidence_level))
    elif method == 'parametric':
        # 파라메트릭 VaR: 정규 분포 가정
        mean = returns.mean()
        std = returns.std()
        from scipy import stats
        z_score = stats.norm.ppf(1 - confidence_level)
        var = abs(mean + z_score * std)
    else:
        raise ValueError(f"알 수 없는 VaR 방식: {method}")
    
    return float(var)


def calculate_cvar(returns: pd.Series, confidence_level: float = 0.95) -> float:
    """
    CVaR(Conditional Value at Risk) / 예상 손실(Expected Shortfall)을 계산합니다.
    
    Args:
        returns: 수익률 시리즈
        confidence_level: 신뢰 수준 (예: 0.95 = 95%)
    
    Returns:
        CVaR (양수)
    """
    if len(returns) == 0:
        return 0.0
    
    var_threshold = returns.quantile(1 - confidence_level)
    cvar = abs(returns[returns <= var_threshold].mean())
    
    return float(cvar)


def get_risk_metrics(daily_returns: pd.Series, equity_curve: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    종합적인 리스크 지표를 계산합니다.
    
    Args:
        daily_returns: 일일 수익률 시리즈
        equity_curve: 포트폴리오 가치 시리즈
        risk_free_rate: 연간 무위험 이자율
    
    Returns:
        리스크 지표 딕셔너리
    """
    metrics = {
        'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0,
        'annualized_return': calc_annualized_return(daily_returns),
        'annualized_volatility': calc_annualized_volatility(daily_returns),
        'sharpe_ratio': calc_sharpe_ratio(daily_returns, risk_free_rate),
        'sortino_ratio': calc_sortino_ratio(daily_returns, risk_free_rate),
        'max_drawdown': calc_max_drawdown(equity_curve),
        'calmar_ratio': calc_calmar_ratio(daily_returns, equity_curve),
        'var_95': calculate_var(daily_returns, 0.95),
        'cvar_95': calculate_cvar(daily_returns, 0.95),
        'win_rate': (daily_returns > 0).sum() / len(daily_returns) if len(daily_returns) > 0 else 0.0,
        'avg_win': daily_returns[daily_returns > 0].mean() if (daily_returns > 0).any() else 0.0,
        'avg_loss': daily_returns[daily_returns < 0].mean() if (daily_returns < 0).any() else 0.0,
    }
    
    return metrics


if __name__ == "__main__":
    print("=== 리스크 관리 테스트 ===\n")
    
    # 샘플 수익률 생성
    np.random.seed(42)
    n_days = 252
    daily_returns = pd.Series(np.random.normal(0.001, 0.02, n_days))
    
    # 자산 곡선 생성
    equity_curve = (1 + daily_returns).cumprod() * 100000
    
    print("샘플 데이터 생성됨:")
    print(f"  일수: {n_days}")
    print(f"  초기 자본: $100,000")
    
    # 지표 계산
    print("\n--- 리스크 지표 ---")
    metrics = get_risk_metrics(daily_returns, equity_curve)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            if 'return' in key or 'ratio' in key:
                print(f"{key}: {value:.4f}")
            elif 'rate' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.4f}")
    
    # 변동성 타겟팅 테스트
    print("\n--- 변동성 타겟팅 ---")
    target_vol = 0.15
    realized_vol = calc_annualized_volatility(daily_returns)
    position_size = volatility_target_position_size(target_vol, realized_vol)
    
    print(f"목표 변동성: {target_vol:.2%}")
    print(f"실현 변동성: {realized_vol:.2%}")
    print(f"제안 포지션 크기: {position_size:.2f}x")
    
    # 낙폭 제한 테스트
    print("\n--- 낙폭 제한 테스트 ---")
    max_dd = calc_max_drawdown(equity_curve)
    print(f"현재 최대 낙폭: {max_dd:.2%}")
    
    limited_equity = apply_drawdown_limit(equity_curve, max_dd_limit=0.15)
    limited_returns = limited_equity.pct_change().fillna(0)
    limited_max_dd = calc_max_drawdown(limited_equity)
    
    print(f"15% 제한 적용 후 최대 낙폭: {limited_max_dd:.2%}")
    print(f"최종 자산 (원본): ${equity_curve.iloc[-1]:,.2f}")
    print(f"최종 자산 (제한됨): ${limited_equity.iloc[-1]:,.2f}")
