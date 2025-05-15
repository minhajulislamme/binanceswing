import logging
import numpy as np
import pandas as pd
import ta
import math
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class SupertrendIndicator:
    """Supertrend indicator implementation for faster trend detection"""
    def __init__(self, period=10, multiplier=3.0):
        self.period = period
        self.multiplier = multiplier
        
    def calculate(self, df):
        """
        Calculate Supertrend indicator
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with supertrend values and direction
        """
        try:
            # Check if we have enough data
            if len(df) < self.period + 2:
                logger.warning(f"Not enough data for Supertrend calculation. Need {self.period + 2} rows, got {len(df)}")
                # Add empty columns and return
                df['supertrend'] = np.nan
                df['supertrend_direction'] = np.nan
                df['final_upper'] = np.nan
                df['final_lower'] = np.nan
                return df
                
            # Make a copy to avoid modifying original dataframe
            df = df.copy()
            
            # Calculate ATR
            df['atr'] = ta.volatility.average_true_range(
                df['high'], df['low'], df['close'], window=self.period
            )
            
            # Calculate basic upper and lower bands
            df['basic_upper'] = (df['high'] + df['low']) / 2 + (self.multiplier * df['atr'])
            df['basic_lower'] = (df['high'] + df['low']) / 2 - (self.multiplier * df['atr'])
            
            # Initialize Supertrend columns
            df['supertrend'] = np.nan
            df['supertrend_direction'] = np.nan
            df['final_upper'] = np.nan
            df['final_lower'] = np.nan
            
            # Use vectorized operations as much as possible
            # But some operations need row-by-row calculation
            for i in range(self.period, len(df)):
                if i == self.period:
                    # Using .loc to properly set values
                    df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                    df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                    
                    # Initial trend direction
                    if df['close'].iloc[i] <= df['final_upper'].iloc[i]:
                        df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                    else:
                        df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                else:
                    # Calculate upper band
                    if (df['basic_upper'].iloc[i] < df['final_upper'].iloc[i-1] or 
                        df['close'].iloc[i-1] > df['final_upper'].iloc[i-1]):
                        df.loc[df.index[i], 'final_upper'] = df['basic_upper'].iloc[i]
                    else:
                        df.loc[df.index[i], 'final_upper'] = df['final_upper'].iloc[i-1]
                    
                    # Calculate lower band
                    if (df['basic_lower'].iloc[i] > df['final_lower'].iloc[i-1] or 
                        df['close'].iloc[i-1] < df['final_lower'].iloc[i-1]):
                        df.loc[df.index[i], 'final_lower'] = df['basic_lower'].iloc[i]
                    else:
                        df.loc[df.index[i], 'final_lower'] = df['final_lower'].iloc[i-1]
                    
                    # Calculate Supertrend value
                    if (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                        df['close'].iloc[i] <= df['final_upper'].iloc[i]):
                        df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
                    elif (df['supertrend'].iloc[i-1] == df['final_upper'].iloc[i-1] and 
                          df['close'].iloc[i] > df['final_upper'].iloc[i]):
                        df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                    elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                          df['close'].iloc[i] >= df['final_lower'].iloc[i]):
                        df.loc[df.index[i], 'supertrend'] = df['final_lower'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = 1  # Uptrend
                    elif (df['supertrend'].iloc[i-1] == df['final_lower'].iloc[i-1] and 
                          df['close'].iloc[i] < df['final_lower'].iloc[i]):
                        df.loc[df.index[i], 'supertrend'] = df['final_upper'].iloc[i]
                        df.loc[df.index[i], 'supertrend_direction'] = -1  # Downtrend
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating Supertrend: {e}")
            # Add empty columns and return
            df['supertrend'] = np.nan
            df['supertrend_direction'] = np.nan
            df['final_upper'] = np.nan
            df['final_lower'] = np.nan
            return df

class TradingStrategy:
    """Base class for trading strategies"""
    def __init__(self, strategy_name):
        self.strategy_name = strategy_name
        self.risk_manager = None
        # Add cache related attributes
        self._cache = {}
        self._max_cache_entries = 10  # Limit cache size
        self._cache_expiry = 3600  # Cache expiry in seconds (1 hour)
        self._last_kline_time = None
        self._cached_dataframe = None
        
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Convert timestamps to datetime
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Ensure dataframe is sorted by time
        df = df.sort_values('open_time', ascending=True).reset_index(drop=True)
        
        return df
    
    def set_risk_manager(self, risk_manager):
        """Set the risk manager for the strategy"""
        self.risk_manager = risk_manager
        logger.info(f"Risk manager set for {self.strategy_name} strategy")
    
    def get_signal(self, klines):
        """
        Should be implemented by subclasses.
        Returns 'BUY', 'SELL', or None.
        """
        raise NotImplementedError("Each strategy must implement get_signal method")

class SwingTraderPro(TradingStrategy):
    """
    SwingTraderPro - An advanced swing trading strategy that identifies key market
    turning points and momentum based on multiple technical indicators, trend strength,
    volatility, and price action patterns.
    
    Features:
    - Trend identification using multiple timeframe analysis
    - Price action pattern recognition for swing entries and exits
    - Dynamic support/resistance level identification
    - Adaptive position sizing based on volatility and trend strength
    - Multiple confirmation signals for trade entries
    - Volatility-based trailing stops and take profit targets
    - Divergence detection for trend exhaustion and reversals
    - Volume profile analysis for high probability trade setups
    - Market regime detection and strategy adjustment
    """
    def __init__(self,
                 ema_fast=8,
                 ema_slow=21,
                 ema_trend=50,
                 macd_fast=12,
                 macd_slow=26,
                 macd_signal=9,
                 rsi_period=14,
                 rsi_overbought=70,
                 rsi_oversold=30,
                 rsi_trend_period=5,
                 stoch_k_period=14,
                 stoch_d_period=3,
                 stoch_overbought=80,
                 stoch_oversold=20,
                 volume_ma_period=20,
                 atr_period=14,
                 atr_multiplier_sl=2.0,
                 atr_multiplier_tp=3.0,
                 supertrend_period=10,
                 supertrend_multiplier=3.0,
                 fibonacci_levels=[0.236, 0.382, 0.5, 0.618, 0.786],
                 adx_period=14,
                 adx_threshold=25,
                 cooloff_period=3,
                 max_consecutive_losses=2):
        
        super().__init__('SwingTraderPro')
        
        # EMA settings
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.ema_trend = ema_trend
        
        # MACD settings
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        
        # RSI settings
        self.rsi_period = rsi_period
        self.rsi_overbought = rsi_overbought
        self.rsi_oversold = rsi_oversold
        self.rsi_trend_period = rsi_trend_period
        
        # Stochastic settings
        self.stoch_k_period = stoch_k_period
        self.stoch_d_period = stoch_d_period
        self.stoch_overbought = stoch_overbought
        self.stoch_oversold = stoch_oversold
        
        # Volume and volatility settings
        self.volume_ma_period = volume_ma_period
        self.atr_period = atr_period
        self.atr_multiplier_sl = atr_multiplier_sl
        self.atr_multiplier_tp = atr_multiplier_tp
        
        # Additional indicators
        self.supertrend_period = supertrend_period
        self.supertrend_multiplier = supertrend_multiplier
        self.fibonacci_levels = fibonacci_levels
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        
        # Risk management settings
        self.cooloff_period = cooloff_period
        self.max_consecutive_losses = max_consecutive_losses
        
        # State variables
        self.consecutive_losses = 0
        self.last_loss_time = None
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        self.current_market_condition = None
        self.current_trend = None
        
        # Initialize Supertrend indicator
        self.supertrend_indicator = SupertrendIndicator(
            period=self.supertrend_period,
            multiplier=self.supertrend_multiplier
        )
    
    def prepare_data(self, klines):
        """Convert raw klines to a DataFrame with OHLCV data"""
        # Use the base class implementation
        df = super().prepare_data(klines)
        return df
    
    def add_indicators(self, df):
        """Add technical indicators to the DataFrame for swing trading strategy"""
        if len(df) < self.ema_trend + 10:
            return df  # Not enough data for reliable indicators
        
        # Calculate basic moving averages for trend identification
        df['ema_fast'] = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
        df['ema_slow'] = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
        df['ema_trend'] = ta.trend.ema_indicator(df['close'], window=self.ema_trend)
        
        # MACD for momentum
        macd = ta.trend.MACD(df['close'], 
                            window_fast=self.macd_fast, 
                            window_slow=self.macd_slow, 
                            window_sign=self.macd_signal)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # Add MACD crossover signal: 1 for bullish crossover, -1 for bearish, 0 for no crossover
        df['macd_crossover'] = 0
        for i in range(1, len(df)):
            if df['macd'].iloc[i] > df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] <= df['macd_signal'].iloc[i-1]:
                df.loc[i, 'macd_crossover'] = 1  # Bullish crossover
            elif df['macd'].iloc[i] < df['macd_signal'].iloc[i] and df['macd'].iloc[i-1] >= df['macd_signal'].iloc[i-1]:
                df.loc[i, 'macd_crossover'] = -1  # Bearish crossover
        
        # Calculate RSI for overbought/oversold conditions
        df['rsi'] = ta.momentum.rsi(df['close'], window=self.rsi_period)
        
        # Calculate RSI trend (average of recent RSI values)
        df['rsi_trend'] = ta.trend.sma_indicator(df['rsi'], window=self.rsi_trend_period)
        
        # Volume analysis
        df['volume_ma'] = ta.trend.sma_indicator(df['volume'], window=self.volume_ma_period)
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Calculate Volume-weighted RSI
        df['volume_weighted_rsi'] = df['rsi'] * df['volume_ratio']
        
        # Stochastic oscillator
        stoch = ta.momentum.StochasticOscillator(
            df['high'], df['low'], df['close'], 
            window=self.stoch_k_period,
            smooth_window=self.stoch_d_period
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR for volatility measurement and stop loss calculation
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        df['atr_pct'] = (df['atr'] / df['close']) * 100  # ATR as percentage of price
        
        # Supertrend indicator for trend direction
        df = self.supertrend_indicator.calculate(df)
        
        # Bollinger Bands for volatility and potential reversals
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX for trend strength
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.adx_period)
        df['adx'] = adx.adx()
        df['di_plus'] = adx.adx_pos()
        df['di_minus'] = adx.adx_neg()
        
        # Calculate trend based on multiple indicators
        df['trend'] = self.determine_trend(df)
        
        # Identify market conditions (ranging, trending, volatile)
        df['market_condition'] = self.classify_market_condition(df)
        
        # Calculate Fibonacci levels based on recent swing highs and lows
        self.calculate_fibonacci_levels(df)
        
        # Detect potential reversal patterns
        df['potential_reversal'] = self.detect_reversal_patterns(df)
        
        # Calculate VWAP for intraday reference
        df['vwap'] = self.calculate_vwap(df)
        
        return df
    
    def determine_trend(self, df):
        """Determine the current trend based on multiple indicators"""
        trend = pd.Series(index=df.index, dtype='object')
        
        for i in range(len(df)):
            if i < self.ema_trend:  # Not enough data yet
                trend.loc[i] = 'UNKNOWN'
                continue
                
            # Price relative to EMAs
            price_above_trend_ema = df['close'].iloc[i] > df['ema_trend'].iloc[i]
            price_above_slow_ema = df['close'].iloc[i] > df['ema_slow'].iloc[i]
            fast_above_slow = df['ema_fast'].iloc[i] > df['ema_slow'].iloc[i]
            
            # Consider supertrend if available
            supertrend_bullish = False
            if i >= self.supertrend_period:
                supertrend_bullish = df['supertrend_direction'].iloc[i] == 1
            
            # MACD histogram
            macd_hist_positive = df['macd_hist'].iloc[i] > 0
            
            # Determine trend based on combined signals
            if price_above_trend_ema and price_above_slow_ema and fast_above_slow and (supertrend_bullish or macd_hist_positive):
                trend.loc[i] = 'UPTREND'
            elif not price_above_trend_ema and not price_above_slow_ema and not fast_above_slow and (not supertrend_bullish or not macd_hist_positive):
                trend.loc[i] = 'DOWNTREND'
            else:
                # Check if currently in a pullback within a larger trend
                if i > 0 and trend.iloc[i-1] == 'UPTREND' and fast_above_slow:
                    trend.loc[i] = 'UPTREND_PULLBACK'
                elif i > 0 and trend.iloc[i-1] == 'DOWNTREND' and not fast_above_slow:
                    trend.loc[i] = 'DOWNTREND_PULLBACK'
                else:
                    trend.loc[i] = 'SIDEWAYS'
                    
        return trend
    
    def classify_market_condition(self, df):
        """
        Classify market conditions into different categories:
        BULLISH, BEARISH, EXTREME_BULLISH, EXTREME_BEARISH, SIDEWAYS, SQUEEZE
        """
        conditions = []
        
        for i in range(len(df)):
            if i < self.adx_period:
                conditions.append('SIDEWAYS')  # Default for initial rows
                continue
                
            # Get relevant indicators
            adx = df['adx'].iloc[i]
            di_plus = df['di_plus'].iloc[i]
            di_minus = df['di_minus'].iloc[i]
            rsi = df['rsi'].iloc[i]
            bb_width = df['bb_width'].iloc[i]
            supertrend_dir = df['supertrend_direction'].iloc[i] if i >= self.supertrend_period else 0
            macd_crossover = df['macd_crossover'].iloc[i] if 'macd_crossover' in df else 0
            
            # Check for squeeze condition (low volatility, potential breakout)
            is_squeeze = bb_width < 0.5  # Threshold adjusted for swing trading
            
            # Enhanced condition classification with multi-indicator confirmation
            if is_squeeze:
                conditions.append('SQUEEZE')
            else:
                # Strong bullish trend confirmation
                if (adx > self.adx_threshold and 
                    di_plus > di_minus and 
                    supertrend_dir > 0 and
                    (rsi > 50 or macd_crossover > 0)):
                    conditions.append('BULLISH')
                    
                # Strong bearish trend confirmation
                elif (adx > self.adx_threshold and 
                      di_minus > di_plus and 
                      supertrend_dir < 0 and
                      (rsi < 50 or macd_crossover < 0)):
                    conditions.append('BEARISH')
                    
                # Extreme bullish trend
                elif (adx > self.adx_threshold * 1.5 and 
                      di_plus > di_minus * 1.5 and 
                      supertrend_dir > 0):
                    conditions.append('EXTREME_BULLISH')
                    
                # Extreme bearish trend
                elif (adx > self.adx_threshold * 1.5 and 
                      di_minus > di_plus * 1.5 and 
                      supertrend_dir < 0):
                    conditions.append('EXTREME_BEARISH')
                    
                # Default to sideways when no strong trend is detected
                else:
                    conditions.append('SIDEWAYS')
        
        return pd.Series(conditions, index=df.index)
    
    def calculate_vwap(self, df):
        """Calculate VWAP (Volume Weighted Average Price)"""
        # Get date component of timestamp for grouping
        df['date'] = df['open_time'].dt.date
        
        # Calculate VWAP for each day
        vwap = pd.Series(index=df.index)
        for date, group in df.groupby('date'):
            # Calculate cumulative sum of price * volume
            cum_vol_price = (group['close'] * group['volume']).cumsum()
            # Calculate cumulative sum of volume
            cum_vol = group['volume'].cumsum()
            # Calculate VWAP
            daily_vwap = cum_vol_price / cum_vol
            # Add to result series
            vwap.loc[group.index] = daily_vwap
            
        return vwap
    
    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement/extension levels for support and resistance"""
        if len(df) < 20:  # Need sufficient data
            return
        
        # Find recent swing high and low points
        window = min(100, len(df) - 1)  # Look back window
        price_data = df['close'].iloc[-window:]
        
        # Identify swing high and low
        swing_high = price_data.max()
        swing_low = price_data.min()
        
        # Reset fibonacci levels
        self.fib_support_levels = []
        self.fib_resistance_levels = []
        
        # Calculate levels based on trend
        latest = df.iloc[-1]
        current_price = latest['close']
        current_trend = latest['trend']
        
        if current_trend in ['UPTREND', 'UPTREND_PULLBACK']:
            # In uptrend, calculate fib retracements from low to high for support
            for fib in self.fibonacci_levels:
                level = swing_low + (swing_high - swing_low) * (1 - fib)
                if level < current_price:
                    self.fib_support_levels.append(level)
                else:
                    self.fib_resistance_levels.append(level)
                    
            # Add extension levels for resistance
            for ext in [1.272, 1.618, 2.0]:
                level = swing_low + (swing_high - swing_low) * ext
                self.fib_resistance_levels.append(level)
                
        else:  # DOWNTREND or DOWNTREND_PULLBACK
            # In downtrend, calculate fib retracements from high to low for resistance
            for fib in self.fibonacci_levels:
                level = swing_high - (swing_high - swing_low) * fib
                if level > current_price:
                    self.fib_resistance_levels.append(level)
                else:
                    self.fib_support_levels.append(level)
                    
            # Add extension levels for support
            for ext in [1.272, 1.618, 2.0]:
                level = swing_high - (swing_high - swing_low) * ext
                self.fib_support_levels.append(level)
        
        # Sort the levels
        self.fib_support_levels.sort(reverse=True)  # Descending
        self.fib_resistance_levels.sort()  # Ascending
    
    def detect_reversal_patterns(self, df):
        """
        Advanced reversal pattern detection for swing trading
        Returns 1 for potential bullish reversal, -1 for bearish reversal, 0 for no reversal
        """
        if len(df) < 5:
            return pd.Series(0, index=df.index)
            
        # Initialize result series
        reversal = pd.Series(0, index=df.index)
        
        for i in range(4, len(df)):
            # Get relevant rows for pattern detection
            curr = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            prev3 = df.iloc[i-3]
            
            # Check for bullish reversal patterns
            bullish_reversal = False
            
            # Bullish engulfing
            if (curr['close'] > curr['open'] and  # Current candle is bullish
                prev1['close'] < prev1['open'] and  # Previous candle is bearish
                curr['close'] > prev1['open'] and  # Current close above prev open
                curr['open'] < prev1['close']):  # Current open below prev close
                bullish_reversal = True
                
            # Hammer pattern (bullish)
            elif (curr['low'] < curr['open'] and
                  curr['low'] < curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) < 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (min(curr['open'], curr['close']) - curr['low']) > 
                  (curr['high'] - max(curr['open'], curr['close'])) * 3):
                bullish_reversal = True
                
            # RSI divergence (bullish)
            elif (prev2['low'] > prev1['low'] and  # Price making lower low
                  prev2['rsi'] < prev1['rsi'] and  # RSI making higher low
                  curr['supertrend_direction'] == 1):  # Confirmed by Supertrend
                bullish_reversal = True
                
            # Check for bearish reversal patterns
            bearish_reversal = False
            
            # Bearish engulfing
            if (curr['close'] < curr['open'] and  # Current candle is bearish
                prev1['close'] > prev1['open'] and  # Previous candle is bullish
                curr['close'] < prev1['open'] and  # Current close below prev open
                curr['open'] > prev1['close']):  # Current open above prev close
                bearish_reversal = True
                
            # Shooting star (bearish)
            elif (curr['high'] > curr['open'] and
                  curr['high'] > curr['close'] and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 2 and
                  (curr['high'] - max(curr['open'], curr['close'])) > 
                  (min(curr['open'], curr['close']) - curr['low']) * 3):
                bearish_reversal = True
                
            # RSI divergence (bearish)
            elif (prev2['high'] < prev1['high'] and  # Price making higher high
                  prev2['rsi'] > prev1['rsi'] and  # RSI making lower high
                  curr['supertrend_direction'] == -1):  # Confirmed by Supertrend
                bearish_reversal = True
            
            # Set reversal value
            if bullish_reversal:
                reversal.loc[i] = 1
            elif bearish_reversal:
                reversal.loc[i] = -1
                
        return reversal
        
    def detect_support_resistance_levels(self, df):
        """Detect key support and resistance levels from price action"""
        levels = []
        window = min(200, len(df) - 1)
        
        # Use price action to identify significant levels
        price_data = df['close'].iloc[-window:]
        
        # Find local minima and maxima
        for i in range(5, len(price_data) - 5):
            # Local maxima: price higher than surrounding prices
            if (price_data.iloc[i] > price_data.iloc[i-1] and 
                price_data.iloc[i] > price_data.iloc[i-2] and
                price_data.iloc[i] > price_data.iloc[i+1] and
                price_data.iloc[i] > price_data.iloc[i+2]):
                levels.append({'price': price_data.iloc[i], 'type': 'resistance'})
                
            # Local minima: price lower than surrounding prices
            if (price_data.iloc[i] < price_data.iloc[i-1] and 
                price_data.iloc[i] < price_data.iloc[i-2] and
                price_data.iloc[i] < price_data.iloc[i+1] and
                price_data.iloc[i] < price_data.iloc[i+2]):
                levels.append({'price': price_data.iloc[i], 'type': 'support'})
                
        return levels
        
    def in_cooloff_period(self, current_time):
        """Check if we're in a cool-off period after consecutive losses"""
        if self.consecutive_losses >= self.max_consecutive_losses and self.last_loss_time:
            # Check if enough time has passed since last loss
            if isinstance(current_time, pd.Timestamp) and isinstance(self.last_loss_time, datetime):
                cutoff_time = self.last_loss_time + timedelta(hours=self.cooloff_period)
                return pd.Timestamp(cutoff_time) > current_time
                
        return False
        
    def update_trade_result(self, was_profitable):
        """
        Update consecutive losses counter for cool-off period calculation
        
        Args:
            was_profitable: Boolean indicating if the last trade was profitable
        """
        if was_profitable:
            # Reset consecutive losses on profitable trade
            self.consecutive_losses = 0
            self.last_loss_time = None
            logger.info("Profitable trade - reset consecutive losses counter")
        else:
            # Increment consecutive losses
            self.consecutive_losses += 1
            self.last_loss_time = datetime.now()
            logger.info(f"Loss recorded - consecutive losses: {self.consecutive_losses}")
            
            # Check if we need to enter cool-off period
            if self.consecutive_losses >= self.max_consecutive_losses:
                logger.info(f"Entering cool-off period for {self.cooloff_period} candles")
    
    def detect_divergence(self, df):
        """Detect price and oscillator divergences for potential reversals"""
        if len(df) < 10:
            return None
            
        # Check last 10 candles for divergence
        window = min(30, len(df) - 1)
        price_data = df[['close', 'high', 'low', 'rsi', 'macd']].iloc[-window:]
        
        # Initialize
        bullish_div = False
        bearish_div = False
        
        # Find local minima and maxima in price
        min_indices = []
        max_indices = []
        for i in range(2, len(price_data) - 2):
            # Local minima
            if (price_data['low'].iloc[i] < price_data['low'].iloc[i-1] and
                price_data['low'].iloc[i] < price_data['low'].iloc[i-2] and
                price_data['low'].iloc[i] < price_data['low'].iloc[i+1] and
                price_data['low'].iloc[i] < price_data['low'].iloc[i+2]):
                min_indices.append(i)
                
            # Local maxima
            if (price_data['high'].iloc[i] > price_data['high'].iloc[i-1] and
                price_data['high'].iloc[i] > price_data['high'].iloc[i-2] and
                price_data['high'].iloc[i] > price_data['high'].iloc[i+1] and
                price_data['high'].iloc[i] > price_data['high'].iloc[i+2]):
                max_indices.append(i)
        
        # Check for bullish divergence (price making lower lows, oscillator making higher lows)
        if len(min_indices) >= 2:
            p1, p2 = min_indices[-2], min_indices[-1]
            if price_data['low'].iloc[p2] < price_data['low'].iloc[p1]:
                # RSI divergence
                if price_data['rsi'].iloc[p2] > price_data['rsi'].iloc[p1]:
                    bullish_div = True
                # MACD divergence
                elif price_data['macd'].iloc[p2] > price_data['macd'].iloc[p1]:
                    bullish_div = True
            
        # Check for bearish divergence (price making higher highs, oscillator making lower highs)
        if len(max_indices) >= 2:
            p1, p2 = max_indices[-2], max_indices[-1]
            if price_data['high'].iloc[p2] > price_data['high'].iloc[p1]:
                # RSI divergence
                if price_data['rsi'].iloc[p2] < price_data['rsi'].iloc[p1]:
                    bearish_div = True
                # MACD divergence
                elif price_data['macd'].iloc[p2] < price_data['macd'].iloc[p1]:
                    bearish_div = True
        
        if bullish_div:
            return 'BULLISH'
        elif bearish_div:
            return 'BEARISH'
        else:
            return None
    
    def check_swing_entry(self, df):
        """
        Check for swing trade entry signals based on multi-indicator confirmation
        """
        if len(df) < 10:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Extract key indicators
        trend = latest['trend']
        supertrend_dir = latest['supertrend_direction']
        rsi = latest['rsi']
        stoch_k = latest['stoch_k']
        stoch_d = latest['stoch_d']
        macd = latest['macd']
        macd_signal = latest['macd_signal']
        macd_hist = latest['macd_hist']
        market_condition = latest['market_condition']
        potential_reversal = latest['potential_reversal']
        
        # Check for bullish entry
        if (trend in ['UPTREND', 'UPTREND_PULLBACK', 'SIDEWAYS'] and 
            supertrend_dir == 1):
            
            # Entry on RSI crossing up from oversold
            if rsi > 30 and df['rsi'].iloc[-2] <= 30:
                return 'BUY'
                
            # Entry on stochastic crossover from oversold
            if (stoch_k > stoch_d and 
                df['stoch_k'].iloc[-2] <= df['stoch_d'].iloc[-2] and
                stoch_k < 30):
                return 'BUY'
                
            # Entry on MACD crossover
            if (macd > macd_signal and 
                df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]):
                return 'BUY'
                
            # Entry at support level (Fibonacci or price action)
            close_to_fib_support = any(abs(latest['close'] - level) / latest['close'] < 0.01 for level in self.fib_support_levels)
            if close_to_fib_support:
                return 'BUY'
                
            # Strong bullish reversal pattern
            if potential_reversal == 1:
                return 'BUY'
        
        # Check for bearish entry
        if (trend in ['DOWNTREND', 'DOWNTREND_PULLBACK', 'SIDEWAYS'] and 
            supertrend_dir == -1):
            
            # Entry on RSI crossing down from overbought
            if rsi < 70 and df['rsi'].iloc[-2] >= 70:
                return 'SELL'
                
            # Entry on stochastic crossover from overbought
            if (stoch_k < stoch_d and 
                df['stoch_k'].iloc[-2] >= df['stoch_d'].iloc[-2] and
                stoch_k > 70):
                return 'SELL'
                
            # Entry on MACD crossover
            if (macd < macd_signal and 
                df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]):
                return 'SELL'
                
            # Entry at resistance level (Fibonacci or price action)
            close_to_fib_resistance = any(abs(latest['close'] - level) / latest['close'] < 0.01 for level in self.fib_resistance_levels)
            if close_to_fib_resistance:
                return 'SELL'
                
            # Strong bearish reversal pattern
            if potential_reversal == -1:
                return 'SELL'
        
        # No valid signal
        return None
    
    def get_momentum_signal(self, df):
        """
        Check for momentum-based signals, particularly for continuation of trends
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        
        # In strong trend conditions, look for momentum entries
        if market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            # Look for pullbacks that have completed in an uptrend
            if (latest['trend'] == 'UPTREND_PULLBACK' and 
                latest['close'] > latest['ema_slow'] and
                latest['stoch_k'] > latest['stoch_d'] and
                latest['stoch_k'] > 20):
                return 'BUY'
                
        elif market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            # Look for rallies that have completed in a downtrend
            if (latest['trend'] == 'DOWNTREND_PULLBACK' and 
                latest['close'] < latest['ema_slow'] and
                latest['stoch_k'] < latest['stoch_d'] and
                latest['stoch_k'] < 80):
                return 'SELL'
        
        return None
    
    def get_breakout_signal(self, df):
        """
        Check for breakout signals from consolidation or squeeze patterns
        """
        if len(df) < 10:
            return None
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Only look for breakouts in squeeze or sideways conditions
        if latest['market_condition'] == 'SQUEEZE':
            # Volume spike on potential breakout
            if latest['volume_ratio'] > 1.5:
                # Upside breakout
                if (latest['close'] > latest['bb_upper'] and
                    prev['close'] <= prev['bb_upper'] and
                    latest['supertrend_direction'] == 1):
                    return 'BUY'
                    
                # Downside breakout
                elif (latest['close'] < latest['bb_lower'] and
                    prev['close'] >= prev['bb_lower'] and
                    latest['supertrend_direction'] == -1):
                    return 'SELL'
        
        # Check for breakouts from key levels
        levels = self.detect_support_resistance_levels(df)
        
        for level in levels:
            # Breakout above resistance
            if (level['type'] == 'resistance' and
                prev['close'] < level['price'] and
                latest['close'] > level['price'] and
                latest['volume_ratio'] > 1.3):
                return 'BUY'
                
            # Breakdown below support
            elif (level['type'] == 'support' and
                 prev['close'] > level['price'] and
                 latest['close'] < level['price'] and
                 latest['volume_ratio'] > 1.3):
                return 'SELL'
        
        return None
    
    def get_counter_trend_signal(self, df):
        """
        Look for counter-trend reversal opportunities in extreme conditions
        """
        if len(df) < 5:
            return None
            
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        potential_reversal = latest['potential_reversal']
        
        # Only consider counter-trend trades in extreme conditions
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            # Check for reversal confirmation
            if potential_reversal == 1 and market_condition == 'EXTREME_BEARISH':
                # Confirm with divergence
                divergence = self.detect_divergence(df)
                if divergence == 'BULLISH':
                    return 'BUY'
                    
            elif potential_reversal == -1 and market_condition == 'EXTREME_BULLISH':
                # Confirm with divergence
                divergence = self.detect_divergence(df)
                if divergence == 'BEARISH':
                    return 'SELL'
        
        return None
    
    def get_signal(self, klines):
        """
        Master signal generation method integrating multiple strategies
        """
        # Prepare and add indicators to the data
        df = self.prepare_data(klines)
        df = self.add_indicators(df)
        
        if len(df) < max(self.ema_trend, self.supertrend_period) + 10:
            # Not enough data to generate reliable signals
            return None
        
        # Get latest data
        latest = df.iloc[-1]
        market_condition = latest['market_condition']
        current_time = latest['open_time']
        
        # Update risk manager with current market condition
        if self.risk_manager:
            self.risk_manager.set_market_condition(market_condition)
        
        # Check for cool-off period first
        if self.in_cooloff_period(current_time):
            logger.info(f"In cool-off period after {self.consecutive_losses} consecutive losses. No trading signals.")
            return None
        
        # Store current market condition and trend for reference
        self.current_market_condition = market_condition
        self.current_trend = latest['trend']
        
        # Prioritize signal generation based on market condition and strategy
        
        # 1. Check for counter-trend reversal signals in extreme market conditions
        if market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH']:
            counter_trend = self.get_counter_trend_signal(df)
            if counter_trend:
                logger.info(f"Counter-trend reversal signal: {counter_trend}")
                return counter_trend
        
        # 2. Check for breakout signals in squeeze conditions
        if market_condition == 'SQUEEZE':
            breakout = self.get_breakout_signal(df)
            if breakout:
                logger.info(f"Breakout signal: {breakout}")
                return breakout
                
        # 3. Check for momentum signals in trending markets
        if market_condition in ['BULLISH', 'BEARISH']:
            momentum = self.get_momentum_signal(df)
            if momentum:
                logger.info(f"Momentum signal: {momentum}")
                return momentum
                
        # 4. Standard swing trading entry signals (works in all market conditions)
        swing_entry = self.check_swing_entry(df)
        if swing_entry:
            logger.info(f"Swing entry signal: {swing_entry}")
            return swing_entry
            
        # No valid signal found
        return None



# Update the factory function to include SwingTraderPro strategy
def get_strategy(strategy_name):
    """Factory function to get a strategy by name"""
    from modules.config import (
        SWING_EMA_FAST, SWING_EMA_SLOW, SWING_EMA_TREND, SWING_MACD_FAST, SWING_MACD_SLOW,
        SWING_MACD_SIGNAL, SWING_RSI_PERIOD, SWING_RSI_OVERBOUGHT, SWING_RSI_OVERSOLD,
        SWING_RSI_TREND_PERIOD, SWING_STOCH_K_PERIOD, SWING_STOCH_D_PERIOD,
        SWING_STOCH_OVERBOUGHT, SWING_STOCH_OVERSOLD, SWING_VOLUME_MA_PERIOD,
        SWING_ATR_PERIOD, SWING_ATR_MULTIPLIER_SL, SWING_ATR_MULTIPLIER_TP
    )
    
    # Default to SwingTraderPro if no strategy specified
    if not strategy_name or strategy_name == 'SwingTraderPro':
        return SwingTraderPro(
            ema_fast=SWING_EMA_FAST,
            ema_slow=SWING_EMA_SLOW,
            ema_trend=SWING_EMA_TREND,
            macd_fast=SWING_MACD_FAST,
            macd_slow=SWING_MACD_SLOW,
            macd_signal=SWING_MACD_SIGNAL,
            rsi_period=SWING_RSI_PERIOD,
            rsi_overbought=SWING_RSI_OVERBOUGHT,
            rsi_oversold=SWING_RSI_OVERSOLD,
            rsi_trend_period=SWING_RSI_TREND_PERIOD,
            stoch_k_period=SWING_STOCH_K_PERIOD,
            stoch_d_period=SWING_STOCH_D_PERIOD,
            stoch_overbought=SWING_STOCH_OVERBOUGHT,
            stoch_oversold=SWING_STOCH_OVERSOLD,
            volume_ma_period=SWING_VOLUME_MA_PERIOD,
            atr_period=SWING_ATR_PERIOD,
            atr_multiplier_sl=SWING_ATR_MULTIPLIER_SL,
            atr_multiplier_tp=SWING_ATR_MULTIPLIER_TP,
            supertrend_period=10,
            supertrend_multiplier=3.0
        )
    else:
        logger.warning(f"Unknown strategy: {strategy_name}. Using SwingTraderPro instead.")
        return SwingTraderPro()

def get_strategy_for_symbol(symbol, strategy_name=None):
    """
    Get a strategy instance configured for a specific symbol
    
    Args:
        symbol: Trading pair symbol
        strategy_name: Optional strategy name
        
    Returns:
        TradingStrategy: A strategy instance
    """
    # If no strategy name provided, use the default from config
    from modules.config import STRATEGY
    if not strategy_name:
        strategy_name = STRATEGY
    
    # Get a strategy instance
    strategy = get_strategy(strategy_name)
    
    # Symbol-specific strategy adjustments
    if "RAYSOL" in symbol:
        # More aggressive settings for RAYSOL
        logger.info(f"Adjusting strategy parameters for high volatility token: {symbol}")
        if isinstance(strategy, SwingTraderPro):
            # For SwingTraderPro, adjust ATR multiplier to account for higher volatility
            strategy.atr_multiplier_sl = strategy.atr_multiplier_sl * 1.5
            strategy.atr_multiplier_tp = strategy.atr_multiplier_tp * 1.2
            # Reinitialize the Supertrend indicator with updated parameters
            strategy.supertrend_indicator = SupertrendIndicator(
                period=strategy.supertrend_period,
                multiplier=strategy.supertrend_multiplier * 1.3  # Increase multiplier by 30%
            )
            strategy.supertrend_indicator = SupertrendIndicator(
                period=strategy.supertrend_period,
                multiplier=strategy.supertrend_multiplier * 1.3  # Increase multiplier by 30%
            )
    
    return strategy