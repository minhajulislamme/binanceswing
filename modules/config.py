import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API configuration
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')

# Testnet configuration
API_TESTNET = os.getenv('BINANCE_API_TESTNET', 'False').lower() == 'true'

# API URLs - Automatically determined based on testnet setting
if API_TESTNET:
    # Testnet URLs
    API_URL = 'https://testnet.binancefuture.com'
    WS_BASE_URL = 'wss://stream.binancefuture.com'
else:
    # Production URLs
    API_URL = os.getenv('BINANCE_API_URL', 'https://fapi.binance.com')
    WS_BASE_URL = 'wss://fstream.binance.com'

# API request settings
RECV_WINDOW = int(os.getenv('BINANCE_RECV_WINDOW', '10000'))

# Trading parameters
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
TRADING_TYPE = 'FUTURES'  # Use futures trading
LEVERAGE = int(os.getenv('LEVERAGE', '10'))
MARGIN_TYPE = os.getenv('MARGIN_TYPE', 'ISOLATED')  # ISOLATED or CROSSED
STRATEGY = os.getenv('STRATEGY', 'SwingTraderPro')

# SwingTraderPro settings
SWING_EMA_FAST = int(os.getenv('SWING_EMA_FAST', '8'))
SWING_EMA_SLOW = int(os.getenv('SWING_EMA_SLOW', '21'))
SWING_EMA_TREND = int(os.getenv('SWING_EMA_TREND', '50'))
SWING_MACD_FAST = int(os.getenv('SWING_MACD_FAST', '12'))
SWING_MACD_SLOW = int(os.getenv('SWING_MACD_SLOW', '26'))
SWING_MACD_SIGNAL = int(os.getenv('SWING_MACD_SIGNAL', '9'))
SWING_RSI_PERIOD = int(os.getenv('SWING_RSI_PERIOD', '14'))
SWING_RSI_OVERBOUGHT = int(os.getenv('SWING_RSI_OVERBOUGHT', '70'))
SWING_RSI_OVERSOLD = int(os.getenv('SWING_RSI_OVERSOLD', '30'))
SWING_RSI_TREND_PERIOD = int(os.getenv('SWING_RSI_TREND_PERIOD', '5'))
SWING_STOCH_K_PERIOD = int(os.getenv('SWING_STOCH_K_PERIOD', '14'))
SWING_STOCH_D_PERIOD = int(os.getenv('SWING_STOCH_D_PERIOD', '3'))
SWING_STOCH_OVERBOUGHT = int(os.getenv('SWING_STOCH_OVERBOUGHT', '80'))
SWING_STOCH_OVERSOLD = int(os.getenv('SWING_STOCH_OVERSOLD', '20'))
SWING_VOLUME_MA_PERIOD = int(os.getenv('SWING_VOLUME_MA_PERIOD', '20'))
SWING_ATR_PERIOD = int(os.getenv('SWING_ATR_PERIOD', '14'))
SWING_ATR_MULTIPLIER_SL = float(os.getenv('SWING_ATR_MULTIPLIER_SL', '2.0'))
SWING_ATR_MULTIPLIER_TP = float(os.getenv('SWING_ATR_MULTIPLIER_TP', '3.0'))

# Position sizing
INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '50.0'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.10'))
MAX_OPEN_POSITIONS = int(os.getenv('MAX_OPEN_POSITIONS', '6'))

# Multi-instance configuration for running separate bot instances per trading pair
MULTI_INSTANCE_MODE = os.getenv('MULTI_INSTANCE_MODE', 'True').lower() == 'true'
MAX_POSITIONS_PER_SYMBOL = int(os.getenv('MAX_POSITIONS_PER_SYMBOL', '3'))

# Auto-compounding settings
AUTO_COMPOUND = os.getenv('AUTO_COMPOUND', 'True').lower() == 'true'
COMPOUND_REINVEST_PERCENT = float(os.getenv('COMPOUND_REINVEST_PERCENT', '0.75'))
COMPOUND_INTERVAL = os.getenv('COMPOUND_INTERVAL', 'DAILY')

# Technical indicator parameters
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_OVERBOUGHT = int(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = int(os.getenv('RSI_OVERSOLD', '30'))
FAST_EMA = int(os.getenv('FAST_EMA', '8'))
SLOW_EMA = int(os.getenv('SLOW_EMA', '21'))
TIMEFRAME = os.getenv('TIMEFRAME', '15m')

# Risk management - Standard settings
USE_STOP_LOSS = os.getenv('USE_STOP_LOSS', 'True').lower() == 'true'
STOP_LOSS_PCT = float(os.getenv('STOP_LOSS_PCT', '0.02'))
USE_TAKE_PROFIT = os.getenv('USE_TAKE_PROFIT', 'True').lower() == 'true'
TAKE_PROFIT_PCT = float(os.getenv('TAKE_PROFIT_PCT', '0.05'))
TRAILING_STOP = os.getenv('TRAILING_STOP', 'True').lower() == 'true'
TRAILING_STOP_PCT = float(os.getenv('TRAILING_STOP_PCT', '0.02'))
TRAILING_TAKE_PROFIT = os.getenv('TRAILING_TAKE_PROFIT', 'True').lower() == 'true'
TRAILING_TAKE_PROFIT_PCT = float(os.getenv('TRAILING_TAKE_PROFIT_PCT', '0.03'))

# Adaptive risk management settings for different market conditions
STOP_LOSS_PCT_BULLISH = float(os.getenv('STOP_LOSS_PCT_BULLISH', '0.02'))
STOP_LOSS_PCT_BEARISH = float(os.getenv('STOP_LOSS_PCT_BEARISH', '0.015'))
STOP_LOSS_PCT_SIDEWAYS = float(os.getenv('STOP_LOSS_PCT_SIDEWAYS', '0.01'))

TAKE_PROFIT_PCT_BULLISH = float(os.getenv('TAKE_PROFIT_PCT_BULLISH', '0.06'))
TAKE_PROFIT_PCT_BEARISH = float(os.getenv('TAKE_PROFIT_PCT_BEARISH', '0.04'))
TAKE_PROFIT_PCT_SIDEWAYS = float(os.getenv('TAKE_PROFIT_PCT_SIDEWAYS', '0.03'))

TRAILING_STOP_PCT_BULLISH = float(os.getenv('TRAILING_STOP_PCT_BULLISH', '0.025'))
TRAILING_STOP_PCT_BEARISH = float(os.getenv('TRAILING_STOP_PCT_BEARISH', '0.018'))
TRAILING_STOP_PCT_SIDEWAYS = float(os.getenv('TRAILING_STOP_PCT_SIDEWAYS', '0.012'))

TRAILING_TAKE_PROFIT_PCT_BULLISH = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_BULLISH', '0.035'))
TRAILING_TAKE_PROFIT_PCT_BEARISH = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_BEARISH', '0.025'))
TRAILING_TAKE_PROFIT_PCT_SIDEWAYS = float(os.getenv('TRAILING_TAKE_PROFIT_PCT_SIDEWAYS', '0.018'))

# Safety settings and rate limiting to prevent overloading the API
STOP_LOSS_UPDATE_INTERVAL_SECONDS = 60  # Check and update trailing stops every minute
TAKE_PROFIT_UPDATE_INTERVAL_SECONDS = 60  # Check and update trailing take profits every minute
MAX_ORDER_RETRIES = 3  # Maximum number of retries for order placement
ORDER_RETRY_DELAY_SECONDS = 2  # Wait time between order retries
PRICE_CHECK_TIMEOUT_SECONDS = 10  # Maximum time to wait for price check
RETRY_COUNT = int(os.getenv('RETRY_COUNT', '3'))  # Number of API call retries
RETRY_DELAY = float(os.getenv('RETRY_DELAY', '1.0'))  # Delay between retries in seconds

# Market data settings
MIN_CANDLES_REQUIRED = 50  # Minimum candles needed for strategy calculations
MAX_CANDLES_STORED = 500  # Maximum candles to store in memory
USE_VOLUME_FILTER = True  # Filter out low volume periods

# Emergency settings - these will override other settings in extreme market conditions
EMERGENCY_STOP_LOSS_PCT = 0.1  # 10% emergency stop loss regardless of market condition
MAX_DAILY_LOSS_PCT = 0.2  # Stop trading if daily loss exceeds 20%
VOLATILITY_SCALE_FACTOR = 0.5  # Reduce position size during high volatility

# Backtest settings
BACKTEST_INITIAL_BALANCE = float(os.getenv('BACKTEST_INITIAL_BALANCE', '1000.0'))  # Initial balance for backtest
BACKTEST_COMMISSION = float(os.getenv('BACKTEST_COMMISSION', '0.0004'))  # Commission rate for backtest
BACKTEST_USE_AUTO_COMPOUND = os.getenv('BACKTEST_USE_AUTO_COMPOUND', 'True').lower() == 'true'  # Auto-compound in backtest
BACKTEST_BEFORE_LIVE = os.getenv('BACKTEST_BEFORE_LIVE', 'True').lower() == 'true'  # Run backtest before live trading
BACKTEST_MIN_PROFIT_PCT = float(os.getenv('BACKTEST_MIN_PROFIT_PCT', '5.0'))  # Minimum profit percent to pass validation
BACKTEST_MIN_WIN_RATE = float(os.getenv('BACKTEST_MIN_WIN_RATE', '35.0'))  # Minimum win rate to pass validation
BACKTEST_PERIOD = os.getenv('BACKTEST_PERIOD', '15 days')  # Default backtest period
BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '')  # Custom start date for backtest

# Logging settings
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Notification settings
USE_TELEGRAM = os.getenv('USE_TELEGRAM', 'False').lower() == 'true'
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'False').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '21:00')