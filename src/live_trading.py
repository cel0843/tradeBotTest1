"""
매매 실행을 위한 라이브 트레이딩 모듈입니다 (모의 또는 실전).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Protocol, Optional
from datetime import datetime
from pathlib import Path
import warnings

# 패키지 실행을 위한 절대 임포트
try:
    from src.data_loader import load_or_download, get_latest_price
    from src.model_predict import predict_and_generate_signals, get_latest_signal
    from src.portfolio import equal_weight_portfolio
    from src.config import DATA_DIR, DATA_START, INITIAL_PAPER_CASH, DEFAULT_SYMBOLS, MODEL_DIR
except ImportError:
    # 로컬 실행을 위한 상대 임포트 (fallback)
    from data_loader import load_or_download, get_latest_price
    from model_predict import predict_and_generate_signals, get_latest_signal
    from portfolio import equal_weight_portfolio
    from config import DATA_DIR, DATA_START, INITIAL_PAPER_CASH, DEFAULT_SYMBOLS, MODEL_DIR

warnings.filterwarnings('ignore')


class Broker(Protocol):
    """
    매매 작업을 위한 추상 브로커 인터페이스입니다.
    """
    
    def get_cash(self) -> float:
        """사용 가능한 현금을 가져옵니다."""
        ...
    
    def get_positions(self) -> Dict[str, float]:
        """현재 포지션을 {심볼: 수량} 형태로 가져옵니다."""
        ...
    
    def get_position(self, symbol: str) -> float:
        """특정 심볼의 포지션 크기를 가져옵니다."""
        ...
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None
    ) -> Dict:
        """
        주문을 실행합니다.
        
        Args:
            symbol: 주식 심볼
            side: 'buy' 또는 'sell'
            quantity: 주식 수량
            order_type: 'market' 또는 'limit'
            price: 지정가 주문을 위한 가격
        
        Returns:
            주문 실행 세부 정보
        """
        ...
    
    def get_account_value(self, prices: Dict[str, float]) -> float:
        """총 계좌 가치를 가져옵니다."""
        ...


class PaperBroker:
    """
    시뮬레이션을 위한 모의 투자 브로커입니다.
    """
    
    def __init__(self, initial_cash: float = 100000.0):
        """
        모의 브로커를 초기화합니다.
        
        Args:
            initial_cash: 초기 현금 금액
        """
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.positions: Dict[str, float] = {}
        self.orders_history: List[Dict] = []
        self.trades_history: List[Dict] = []
    
    def get_cash(self) -> float:
        """사용 가능한 현금을 가져옵니다."""
        return self.cash
    
    def get_positions(self) -> Dict[str, float]:
        """현재 포지션을 가져옵니다."""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> float:
        """심볼에 대한 포지션을 가져옵니다."""
        return self.positions.get(symbol, 0.0)
    
    def place_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> Dict:
        """
        주문을 실행합니다 (시뮬레이션 실행).
        
        Args:
            symbol: 주식 심볼
            side: 'buy' 또는 'sell'
            quantity: 주식 수량 (양수)
            order_type: 'market' 또는 'limit'
            price: 지정가 (지정가 주문용)
            current_price: 현재 시장가 (시뮬레이션용)
        
        Returns:
            주문 실행 세부 정보
        """
        if quantity <= 0:
            return {'status': 'rejected', 'reason': '유효하지 않은 수량'}
        
        # 모의 투자의 경우 시장가로 즉시 체결
        if current_price is None:
            # 실제 시나리오에서는 시장 데이터에서 가져옴
            return {'status': 'rejected', 'reason': '가격 정보 없음'}
        
        execution_price = current_price if order_type == 'market' else price
        
        if execution_price is None:
            return {'status': 'rejected', 'reason': '체결 가격 없음'}
        
        # 비용 계산
        if side == 'buy':
            cost = quantity * execution_price
            if cost > self.cash:
                return {'status': 'rejected', 'reason': '현금 부족'}
            
            # 매수 실행
            self.cash -= cost
            self.positions[symbol] = self.positions.get(symbol, 0.0) + quantity
            
        elif side == 'sell':
            current_position = self.positions.get(symbol, 0.0)
            if quantity > current_position:
                return {'status': 'rejected', 'reason': '주식 부족'}
            
            # 매도 실행
            proceeds = quantity * execution_price
            self.cash += proceeds
            self.positions[symbol] = current_position - quantity
            
            # 수량이 0이면 포지션 제거
            if self.positions[symbol] == 0:
                del self.positions[symbol]
        
        else:
            return {'status': 'rejected', 'reason': '유효하지 않은 주문 방향'}
        
        # 주문 기록
        order = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': execution_price,
            'status': 'filled'
        }
        
        self.orders_history.append(order)
        self.trades_history.append(order)
        
        return order
    
    def get_account_value(self, prices: Dict[str, float]) -> float:
        """
        총 계좌 가치를 가져옵니다.
        
        Args:
            prices: 현재 가격 딕셔너리 {심볼: 가격}
        
        Returns:
            총 계좌 가치
        """
        position_value = sum(
            qty * prices.get(symbol, 0.0)
            for symbol, qty in self.positions.items()
        )
        return self.cash + position_value
    
    def get_account_summary(self, prices: Dict[str, float]) -> Dict:
        """
        계좌 요약을 가져옵니다.
        
        Args:
            prices: 현재 가격 딕셔너리
        
        Returns:
            계좌 세부 정보가 담긴 딕셔너리
        """
        total_value = self.get_account_value(prices)
        position_value = total_value - self.cash
        
        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'pnl': total_value - self.initial_cash,
            'return': (total_value / self.initial_cash - 1) if self.initial_cash > 0 else 0.0,
            'positions': self.positions.copy(),
            'n_trades': len(self.trades_history)
        }


def calculate_target_shares(
    symbol: str,
    target_weight: float,
    current_price: float,
    total_value: float,
    current_shares: float = 0.0
) -> float:
    """
    주어진 비중에 대한 목표 주식 수를 계산합니다.
    
    Args:
        symbol: 주식 심볼
        target_weight: 목표 비중 (0-1)
        current_price: 주당 현재 가격
        total_value: 총 포트폴리오 가치
        current_shares: 현재 보유 주식 수
    
    Returns:
        목표 주식 수 (소수점 포함 가능)
    """
    target_value = total_value * target_weight
    target_shares = target_value / current_price if current_price > 0 else 0
    
    return target_shares


def rebalance_to_target_weights(
    broker: Broker,
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    min_trade_value: float = 100.0
) -> List[Dict]:
    """
    목표 비중으로 포트폴리오를 리밸런싱합니다.
    
    Args:
        broker: 브로커 인스턴스
        target_weights: 딕셔너리 {심볼: 비중}
        prices: 딕셔너리 {심볼: 현재가}
        min_trade_value: 실행할 최소 거래 금액
    
    Returns:
        실행된 주문 리스트
    """
    orders = []
    
    # 현재 상태 가져오기
    total_value = broker.get_account_value(prices)
    current_positions = broker.get_positions()
    
    # 각 심볼에 대한 목표 주식 수 계산
    all_symbols = set(target_weights.keys()) | set(current_positions.keys())
    
    for symbol in all_symbols:
        target_weight = target_weights.get(symbol, 0.0)
        current_shares = current_positions.get(symbol, 0.0)
        current_price = prices.get(symbol)
        
        if current_price is None or current_price <= 0:
            print(f"경고: {symbol}에 대한 유효한 가격이 없습니다. 건너뜁니다.")
            continue
        
        # 목표 주식 수 계산
        target_shares = calculate_target_shares(
            symbol, target_weight, current_price, total_value, current_shares
        )
        
        # 차이 계산
        shares_diff = target_shares - current_shares
        trade_value = abs(shares_diff * current_price)
        
        # 차이가 유의미한 경우에만 매매
        if trade_value < min_trade_value:
            continue
        
        # 주문 실행
        if shares_diff > 0:
            # 매수
            order = broker.place_order(
                symbol=symbol,
                side='buy',
                quantity=abs(shares_diff),
                current_price=current_price
            )
        else:
            # 매도
            order = broker.place_order(
                symbol=symbol,
                side='sell',
                quantity=abs(shares_diff),
                current_price=current_price
            )
        
        if order.get('status') == 'filled':
            orders.append(order)
            print(f"  {order['side'].upper()} {symbol} {order['quantity']:.2f}주 @ ${order['price']:.2f}")
        else:
            print(f"  {symbol} 주문 거부됨: {order.get('reason')}")
    
    return orders


def run_daily_trading(
    symbols: List[str],
    broker: Optional[Broker] = None,
    long_threshold: float = 0.55,
    flat_threshold: float = 0.45,
    max_positions: int = 5,
    verbose: bool = True
) -> Dict:
    """
    일일 매매 루틴 실행: 데이터 가져오기, 신호 생성, 매매 실행.
    
    Args:
        symbols: 매매할 심볼 리스트
        broker: 브로커 인스턴스 (None이면 PaperBroker 생성)
        long_threshold: 매수 신호 임계값
        flat_threshold: 관망 신호 임계값
        max_positions: 최대 포지션 수
        verbose: 진행 상황 출력 여부
    
    Returns:
        매매 요약 딕셔너리
    """
    if verbose:
        print("=" * 70)
        print(f"일일 매매 루틴 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
    
    # 브로커가 제공되지 않은 경우 초기화
    if broker is None:
        broker = PaperBroker(initial_cash=INITIAL_PAPER_CASH)
        if verbose:
            print(f"\n초기 자본 ${INITIAL_PAPER_CASH:,.2f}로 모의 브로커 사용")
    
    # 최신 가격 가져오기
    if verbose:
        print(f"\n{len(symbols)}개 심볼에 대한 최신 가격 가져오는 중...")
    
    prices = {}
    for symbol in symbols:
        try:
            price = get_latest_price(symbol)
            prices[symbol] = price
            if verbose:
                print(f"  {symbol}: ${price:.2f}")
        except Exception as e:
            if verbose:
                print(f"  {symbol}: 오류 - {str(e)}")
    
    # 신호 생성
    if verbose:
        print(f"\n매매 신호 생성 중...")
    
    signals_data = {}
    for symbol in symbols:
        if symbol not in prices:
            continue
        
        try:
            signal_info = get_latest_signal(symbol, days=60)
            signals_data[symbol] = signal_info
            
            if verbose:
                signal_name = signal_info['signal_name']
                prob = signal_info['probability']
                print(f"  {symbol}: {signal_name} (확률: {prob:.3f})")
        
        except Exception as e:
            if verbose:
                print(f"  {symbol}: 오류 - {str(e)}")
    
    # 목표 포트폴리오 비중 생성
    if verbose:
        print(f"\n목표 포트폴리오 비중 계산 중...")
    
    # 매수 신호가 있는 심볼 필터링
    buy_signals = {
        symbol: info['probability']
        for symbol, info in signals_data.items()
        if info['signal'] == 1
    }
    
    # 확률순 상위 N개 선택
    if len(buy_signals) > max_positions:
        buy_signals = dict(sorted(buy_signals.items(), key=lambda x: x[1], reverse=True)[:max_positions])
    
    # 선택된 심볼 간 동일 비중
    n_positions = len(buy_signals)
    target_weights = {symbol: 1.0 / n_positions for symbol in buy_signals} if n_positions > 0 else {}
    
    if verbose:
        print(f"  목표 포지션 수: {n_positions}")
        for symbol, weight in target_weights.items():
            print(f"    {symbol}: {weight:.1%}")
    
    # 리밸런싱 실행
    if verbose:
        print(f"\n매매 실행 중...")
    
    orders = rebalance_to_target_weights(broker, target_weights, prices, min_trade_value=100.0)
    
    # 계좌 요약 가져오기
    account = broker.get_account_summary(prices)
    
    if verbose:
        print(f"\n--- 계좌 요약 ---")
        print(f"현금:              ${account['cash']:>15,.2f}")
        print(f"포지션 가치:       ${account['position_value']:>15,.2f}")
        print(f"총 가치:           ${account['total_value']:>15,.2f}")
        print(f"손익(P&L):         ${account['pnl']:>15,.2f}")
        print(f"수익률:            {account['return']:>15.2%}")
        print(f"금일 거래:         {len(orders):>15}")
        print(f"총 거래:           {account['n_trades']:>15}")
        
        print(f"\n--- 현재 포지션 ---")
        if account['positions']:
            for symbol, qty in account['positions'].items():
                price = prices.get(symbol, 0)
                value = qty * price
                weight = value / account['total_value'] if account['total_value'] > 0 else 0
                print(f"  {symbol}: {qty:.2f}주 @ ${price:.2f} = ${value:,.2f} ({weight:.1%})")
        else:
            print("  포지션 없음")
        
        print("=" * 70)
    
    # 요약 반환
    summary = {
        'timestamp': datetime.now(),
        'symbols_analyzed': len(symbols),
        'signals_generated': len(signals_data),
        'target_positions': n_positions,
        'orders_placed': len(orders),
        'account': account,
        'orders': orders,
        'signals': signals_data
    }
    
    return summary


if __name__ == "__main__":
    print("=== 라이브 트레이딩 테스트 ===\n")
    
    # 모델 존재 여부 확인
    available_symbols = []
    for symbol in DEFAULT_SYMBOLS:
        model_path = MODEL_DIR / f"{symbol}_xgb_model.pkl"
        if model_path.exists():
            available_symbols.append(symbol)
    
    if not available_symbols:
        print("학습된 모델을 찾을 수 없습니다. 먼저 모델을 학습하세요:")
        print("  python src/model_train.py")
        print("\n시연을 위해 더미 심볼로 실행합니다...")
        available_symbols = ["AAPL"]
    
    print(f"매매 대상 심볼: {available_symbols}\n")
    
    # 모의 브로커 생성
    broker = PaperBroker(initial_cash=INITIAL_PAPER_CASH)
    
    # 일일 매매 실행
    summary = run_daily_trading(
        symbols=available_symbols,
        broker=broker,
        max_positions=3,
        verbose=True
    )
    
    print("\n✓ 일일 매매 루틴 완료")
    
    # 예시: 다일 시뮬레이션 실행
    print("\n" + "=" * 70)
    print("다일 시뮬레이션")
    print("=" * 70)
    
    print("\n5일 시뮬레이션 실행 중...")
    
    for day in range(5):
        print(f"\n--- {day + 1}일차 ---")
        summary = run_daily_trading(
            symbols=available_symbols,
            broker=broker,
            max_positions=3,
            verbose=False
        )
        
        account = summary['account']
        print(f"총 가치: ${account['total_value']:,.2f} | "
              f"손익: ${account['pnl']:,.2f} ({account['return']:.2%}) | "
              f"거래: {summary['orders_placed']}")
    
    print("\n✓ 다일 시뮬레이션 완료")
