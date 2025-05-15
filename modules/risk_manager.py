import logging
import math
import pandas as pd
import numpy as np
import ta
from datetime import datetime, timedelta
from modules.config import (
    INITIAL_BALANCE, RISK_PER_TRADE, MAX_OPEN_POSITIONS,
    USE_STOP_LOSS, STOP_LOSS_PCT, USE_TAKE_PROFIT, 
    TAKE_PROFIT_PCT, TRAILING_TAKE_PROFIT, TRAILING_TAKE_PROFIT_PCT, TRAILING_STOP, TRAILING_STOP_PCT,
    AUTO_COMPOUND, COMPOUND_REINVEST_PERCENT,
    # Adaptive risk management settings
    STOP_LOSS_PCT_BULLISH, STOP_LOSS_PCT_BEARISH, STOP_LOSS_PCT_SIDEWAYS,
    TAKE_PROFIT_PCT_BULLISH, TAKE_PROFIT_PCT_BEARISH, TAKE_PROFIT_PCT_SIDEWAYS,
    TRAILING_STOP_PCT_BULLISH, TRAILING_STOP_PCT_BEARISH, TRAILING_STOP_PCT_SIDEWAYS,
    TRAILING_TAKE_PROFIT_PCT_BULLISH, TRAILING_TAKE_PROFIT_PCT_BEARISH, TRAILING_TAKE_PROFIT_PCT_SIDEWAYS,
    # Multi-instance mode settings
    MULTI_INSTANCE_MODE, MAX_POSITIONS_PER_SYMBOL
)

logger = logging.getLogger(__name__)

class RiskManager:
    def __init__(self, binance_client):
        """Initialize risk manager with a reference to binance client"""
        self.binance_client = binance_client
        self.initial_balance = None
        self.last_known_balance = None
        self.current_market_condition = None  # Will be set to 'BULLISH', 'BEARISH', or 'SIDEWAYS'
        self.position_size_multiplier = 1.0  # Default position size multiplier
        self.last_trailing_stop_update = {}  # Track when we last updated trailing stops
        self.last_trailing_tp_update = {}    # Track when we last updated trailing take profits
        self.stop_loss_levels = {}           # Track stop loss levels for each symbol
        self.take_profit_levels = {}         # Track take profit levels for each symbol
        
    def set_market_condition(self, market_condition):
        """Set the current market condition for adaptive risk management"""
        valid_conditions = ['BULLISH', 'BEARISH', 'SIDEWAYS', 'EXTREME_BULLISH', 'EXTREME_BEARISH', 'SQUEEZE']
        if market_condition in valid_conditions:
            if self.current_market_condition != market_condition:
                logger.info(f"Market condition changed to {market_condition}")
                self.current_market_condition = market_condition
                
                # When market condition changes significantly, consider updating stop losses
                # and take profits on existing positions
                if (self.current_market_condition in ['EXTREME_BULLISH', 'EXTREME_BEARISH'] and
                    any(pos['position_amount'] != 0 for pos in self.binance_client.get_all_positions())):
                    logger.info("Market condition changed to extreme - will update risk parameters on next check")
        else:
            logger.warning(f"Invalid market condition: {market_condition}. Using default risk parameters.")
            self.current_market_condition = 'SIDEWAYS'  # Default to sideways if invalid
    
    def update_position_sizing(self, position_size_multiplier):
        """
        Update the position size multiplier based on market conditions and volatility
        
        Args:
            position_size_multiplier: A multiplier to adjust position size (0.5 = 50%, 1.0 = 100%, etc.)
        """
        if position_size_multiplier <= 0:
            logger.warning(f"Invalid position size multiplier: {position_size_multiplier}. Using default value of 1.0")
            position_size_multiplier = 1.0
            
        self.position_size_multiplier = position_size_multiplier
        logger.info(f"Position size multiplier updated to {position_size_multiplier:.2f}")
        
    def calculate_position_size(self, symbol, side, price, stop_loss_price=None):
        """
        Calculate position size based on risk parameters
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            price: Current market price
            stop_loss_price: Optional stop loss price for calculating risk
            
        Returns:
            quantity: The position size
        """
        # Get account balance
        balance = self.binance_client.get_account_balance()
        
        # Initialize initial balance if not set
        if self.initial_balance is None:
            self.initial_balance = balance
            self.last_known_balance = balance
            
        # Auto compound logic
        if AUTO_COMPOUND and self.last_known_balance is not None:
            profit = balance - self.last_known_balance
            if profit > 0:
                # We've made profit, apply compounding by increasing risk amount
                logger.info(f"Auto-compounding profit of {profit:.2f} USDT")
                # Update the last known balance
                self.last_known_balance = balance
            
        if balance <= 0:
            logger.error("Insufficient balance to open a position")
            return 0
            
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Could not retrieve symbol info for {symbol}")
            return 0
            
        # Calculate risk amount - use higher risk for small accounts
        small_account = balance < 100.0  # Consider accounts under $100 as small
        
        # Adjust risk per trade for small accounts - more aggressive but safer than no trades
        effective_risk = RISK_PER_TRADE
        if small_account:
            effective_risk = max(RISK_PER_TRADE, 0.05)  # Minimum 5% risk for small accounts
            logger.info(f"Small account detected (${balance:.2f}). Using {effective_risk*100:.1f}% risk per trade.")
        
        # Apply the position size multiplier
        effective_risk = effective_risk * self.position_size_multiplier
        
        # Cap the maximum risk percentage
        max_allowed_risk = 0.25  # Cap at 25% per trade
        if effective_risk > max_allowed_risk:
            logger.warning(f"Risk capped at {max_allowed_risk*100}% (was {effective_risk*100:.1f}%)")
            effective_risk = max_allowed_risk
            
        risk_amount = balance * effective_risk
        
        # Calculate position size based on risk and stop loss
        if stop_loss_price and USE_STOP_LOSS:
            # If stop loss is provided, calculate size based on it
            risk_per_unit = abs(price - stop_loss_price)
            if risk_per_unit <= 0:
                logger.error("Stop loss too close to entry price")
                return 0
                
            # Calculate max quantity based on risk
            max_quantity = risk_amount / risk_per_unit
        else:
            # If no stop loss, use a percentage of balance with leverage
            leverage = self.get_current_leverage(symbol)
            
            # Apply appropriate stop loss percentage based on market condition
            if self.current_market_condition == 'BULLISH':
                sl_pct = STOP_LOSS_PCT_BULLISH
            elif self.current_market_condition == 'BEARISH':
                sl_pct = STOP_LOSS_PCT_BEARISH
            else:  # SIDEWAYS or default
                sl_pct = STOP_LOSS_PCT_SIDEWAYS
                
            # Calculate risk per unit based on implied stop loss
            risk_per_unit = price * sl_pct
            
            # Calculate position size considering leverage
            max_quantity = (risk_amount * leverage) / (price * sl_pct)
        
        # Apply precision to quantity
        quantity_precision = symbol_info['quantity_precision']
        quantity = round_step_size(max_quantity, get_step_size(symbol_info['min_qty']))
        
        # Check minimum notional
        min_notional = symbol_info['min_notional']
        if quantity * price < min_notional:
            logger.warning(f"Position size too small - below minimum notional of {min_notional}")
            
            # For small accounts, force minimum notional even if it exceeds normal risk parameters
            if small_account:
                min_quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
                
                # Make sure we don't use more than 50% of balance for very small accounts
                max_safe_quantity = (balance * 0.5 * leverage) / price
                max_safe_quantity = math.floor(max_safe_quantity * 10**quantity_precision) / 10**quantity_precision
                
                quantity = min(min_quantity, max_safe_quantity)
                
                if quantity * price / leverage > balance * 0.5:
                    logger.warning("Position would use more than 50% of balance - reducing size")
                    quantity = math.floor((balance * 0.5 * leverage / price) * 10**quantity_precision) / 10**quantity_precision
                
                if quantity > 0:
                    logger.info(f"Small account: Adjusted position size to meet minimum notional: {quantity}")
                else:
                    logger.error("Balance too low to open even minimum position")
                    return 0
            else:
                # Normal account handling
                if min_notional / price <= max_quantity:
                    quantity = math.ceil(min_notional / price * 10**quantity_precision) / 10**quantity_precision
                    logger.info(f"Adjusted position size to meet minimum notional: {quantity}")
                else:
                    logger.error(f"Cannot meet minimum notional with current risk settings")
                    return 0
                
        logger.info(f"Calculated position size: {quantity} units at {price} per unit")
        logger.info(f"Position value: {quantity * price:.2f} USDT, Leverage: {leverage}x, Risk: {effective_risk*100:.1f}% of balance")
        
        # Save stop loss and take profit levels for this symbol
        self.stop_loss_levels[symbol] = stop_loss_price if stop_loss_price else self.calculate_stop_loss(symbol, side, price)
        self.take_profit_levels[symbol] = self.calculate_take_profit(symbol, side, price)
        
        return quantity
        
    def get_current_leverage(self, symbol):
        """Get the current leverage for a symbol"""
        position_info = self.binance_client.get_position_info(symbol)
        if position_info:
            return position_info['leverage']
        return 1  # Default to 1x if no position info
        
    def should_open_position(self, symbol):
        """Check if a new position should be opened based on risk rules"""
        # Check if we already have an open position for this symbol
        position_info = self.binance_client.get_position_info(symbol)
        if position_info and abs(position_info['position_amount']) > 0:
            logger.info(f"Already have an open position for {symbol}")
            return False
            
        # Check maximum number of open positions (only for the current trading symbol)
        # This allows separate bot instances for different trading pairs to operate independently
        if MULTI_INSTANCE_MODE:
            # In multi-instance mode, only count positions for the current symbol
            positions = self.binance_client.client.futures_position_information()
            # Check if we've reached the max positions for this symbol
            symbol_positions = [p for p in positions if p['symbol'] == symbol and float(p['positionAmt']) != 0]
            if len(symbol_positions) >= MAX_POSITIONS_PER_SYMBOL:
                logger.info(f"Maximum number of positions for {symbol} ({MAX_POSITIONS_PER_SYMBOL}) reached")
                return False
        else:
            # Original behavior - count all positions
            positions = self.binance_client.client.futures_position_information()
            open_positions = [p for p in positions if float(p['positionAmt']) != 0]
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                logger.info(f"Maximum number of open positions ({MAX_OPEN_POSITIONS}) reached")
                return False
            
        return True
        
    def calculate_stop_loss(self, symbol, side, entry_price):
        """Calculate stop loss price based on configuration and market condition"""
        if not USE_STOP_LOSS:
            return None
            
        # Special handling for high volatility tokens
        is_high_volatility = symbol[-4:] in ["USDT"] and any(token in symbol[:-4] for token in ["SOL", "RAY", "ARB", "DOGE", "SHIB"])
            
        # Choose stop loss percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            stop_loss_pct = STOP_LOSS_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            stop_loss_pct = STOP_LOSS_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            stop_loss_pct = STOP_LOSS_PCT_SIDEWAYS
        else:
            stop_loss_pct = STOP_LOSS_PCT  # Default
            
        # For high volatility tokens, add more buffer to the stop loss
        if is_high_volatility:
            original_pct = stop_loss_pct
            stop_loss_pct = stop_loss_pct * 1.5  # 50% wider stops for high volatility tokens
            logger.info(f"High volatility token detected: Increasing stop loss percentage from {original_pct*100:.2f}% to {stop_loss_pct*100:.2f}%")
            
        if side == "BUY":  # Long position
            stop_price = entry_price * (1 - stop_loss_pct)
        else:  # Short position
            stop_price = entry_price * (1 + stop_loss_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_price = round(stop_price, price_precision)
            
        if is_high_volatility:
            logger.info(f"Calculated high-volatility {self.current_market_condition} stop loss at {stop_price} ({stop_loss_pct*100:.2f}%, enhanced buffer active)")
        else:
            logger.info(f"Calculated {self.current_market_condition} stop loss at {stop_price} ({stop_loss_pct*100:.2f}%)")
        
        # Store the stop loss level for this symbol
        self.stop_loss_levels[symbol] = stop_price
        
        return stop_price
        
    def calculate_take_profit(self, symbol, side, entry_price):
        """Calculate take profit price based on configuration and market condition"""
        if not USE_TAKE_PROFIT:
            return None
            
        # Special handling for high volatility tokens
        is_high_volatility = symbol[-4:] in ["USDT"] and any(token in symbol[:-4] for token in ["SOL", "RAY", "ARB", "DOGE", "SHIB"])
            
        # Choose take profit percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            take_profit_pct = TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            take_profit_pct = TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            take_profit_pct = TAKE_PROFIT_PCT_SIDEWAYS
        else:
            take_profit_pct = TAKE_PROFIT_PCT  # Default
            
        # For high volatility tokens, make take profit more aggressive
        if is_high_volatility:
            original_pct = take_profit_pct
            take_profit_pct = take_profit_pct * 1.2  # 20% wider take profit for high volatility tokens
            logger.info(f"High volatility token detected: Increasing take profit percentage from {original_pct*100:.2f}% to {take_profit_pct*100:.2f}%")
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # Short position
            take_profit_price = entry_price * (1 - take_profit_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        if is_high_volatility:
            logger.info(f"Calculated high-volatility {self.current_market_condition} take profit at {take_profit_price} ({take_profit_pct*100:.2f}%, enhanced setting active)")
        else:
            logger.info(f"Calculated {self.current_market_condition} take profit at {take_profit_price} ({take_profit_pct*100:.2f}%)")
        
        # Store the take profit level for this symbol
        self.take_profit_levels[symbol] = take_profit_price
        
        return take_profit_price
        
    def adjust_stop_loss_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust stop loss for trailing stop functionality
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            current_price: Current market price
            position_info: Optional position info
            
        Returns:
            bool: True if stop loss was adjusted, False otherwise
        """
        if not TRAILING_STOP or symbol not in self.stop_loss_levels:
            return False
            
        # Check if we need to throttle updates (don't update too frequently)
        current_time = datetime.now()
        last_update_time = self.last_trailing_stop_update.get(symbol, None)
        if last_update_time and (current_time - last_update_time).seconds < STOP_LOSS_UPDATE_INTERVAL_SECONDS:
            return False  # Don't update too frequently
            
        # Get position info if not provided
        if not position_info:
            position_info = self.binance_client.get_position_info(symbol)
            if not position_info or float(position_info['position_amount']) == 0:
                return False
        
        current_stop_loss = self.stop_loss_levels[symbol]
        if not current_stop_loss:
            return False
            
        # Choose trailing stop percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            trailing_stop_pct = TRAILING_STOP_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            trailing_stop_pct = TRAILING_STOP_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_stop_pct = TRAILING_STOP_PCT_SIDEWAYS
        else:
            trailing_stop_pct = TRAILING_STOP_PCT  # Default
        
        # Calculate new stop loss level
        if side == "BUY":  # Long position
            # Only move stop loss up for long positions
            new_stop_loss = current_price * (1 - trailing_stop_pct)
            if new_stop_loss <= current_stop_loss:
                return False  # Don't move stop loss down
        else:  # Short position
            # Only move stop loss down for short positions
            new_stop_loss = current_price * (1 + trailing_stop_pct)
            if new_stop_loss >= current_stop_loss:
                return False  # Don't move stop loss up
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop_loss = round(new_stop_loss, price_precision)
            
        # Update the trailing stop
        logger.info(f"Updating trailing stop for {symbol} from {current_stop_loss} to {new_stop_loss}")
        self.stop_loss_levels[symbol] = new_stop_loss
        self.last_trailing_stop_update[symbol] = current_time
        
        # Update the actual stop loss order
        try:
            success = self.binance_client.update_stop_loss(symbol, new_stop_loss)
            if success:
                logger.info(f"Successfully updated trailing stop for {symbol} to {new_stop_loss}")
                return True
            else:
                logger.warning(f"Failed to update trailing stop for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error updating trailing stop: {e}")
            return False
    
    def adjust_take_profit_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust take profit for trailing take profit functionality
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            current_price: Current market price
            position_info: Optional position info
            
        Returns:
            bool: True if take profit was adjusted, False otherwise
        """
        if not TRAILING_TAKE_PROFIT or symbol not in self.take_profit_levels:
            return False
            
        # Check if we need to throttle updates (don't update too frequently)
        current_time = datetime.now()
        last_update_time = self.last_trailing_tp_update.get(symbol, None)
        if last_update_time and (current_time - last_update_time).seconds < TAKE_PROFIT_UPDATE_INTERVAL_SECONDS:
            return False  # Don't update too frequently
            
        # Get position info if not provided
        if not position_info:
            position_info = self.binance_client.get_position_info(symbol)
            if not position_info or float(position_info['position_amount']) == 0:
                return False
        
        current_take_profit = self.take_profit_levels[symbol]
        if not current_take_profit:
            return False
            
        # Choose trailing take profit percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            trailing_tp_pct = TRAILING_TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            trailing_tp_pct = TRAILING_TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_tp_pct = TRAILING_TAKE_PROFIT_PCT_SIDEWAYS
        else:
            trailing_tp_pct = TRAILING_TAKE_PROFIT_PCT  # Default
        
        # Calculate new take profit level
        if side == "BUY":  # Long position
            # Only move take profit up for long positions
            new_take_profit = current_price * (1 + trailing_tp_pct)
            if new_take_profit <= current_take_profit:
                return False  # Don't move take profit down
        else:  # Short position
            # Only move take profit down for short positions
            new_take_profit = current_price * (1 - trailing_tp_pct)
            if new_take_profit >= current_take_profit:
                return False  # Don't move take profit up
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_take_profit = round(new_take_profit, price_precision)
            
        # Update the trailing take profit
        logger.info(f"Updating trailing take profit for {symbol} from {current_take_profit} to {new_take_profit}")
        self.take_profit_levels[symbol] = new_take_profit
        self.last_trailing_tp_update[symbol] = current_time
        
        # Update the actual take profit order
        try:
            success = self.binance_client.update_take_profit(symbol, new_take_profit)
            if success:
                logger.info(f"Successfully updated trailing take profit for {symbol} to {new_take_profit}")
                return True
            else:
                logger.warning(f"Failed to update trailing take profit for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error updating trailing take profit: {e}")
            return False
    
    def update_balance_for_compounding(self):
        """
        Update balance tracking for auto-compounding feature
        This method is called after a profitable trade to adjust risk based on new balance
        
        Returns:
            float: The current account balance
        """
        if not AUTO_COMPOUND:
            return 0.0
            
        # Get current account balance
        current_balance = self.binance_client.get_account_balance()
        
        # Initialize values if not set
        if self.initial_balance is None:
            self.initial_balance = current_balance
            self.last_known_balance = current_balance
            return current_balance
            
        # Calculate profit since last update
        profit = current_balance - self.last_known_balance
        
        if profit > 0:
            # Calculate how much to reinvest based on the compound percentage
            reinvest_amount = profit * COMPOUND_REINVEST_PERCENT
            logger.info(f"Auto-compounding profit: {profit:.2f} USDT, reinvesting {reinvest_amount:.2f} USDT")
            
            # Update the last known balance
            self.last_known_balance = current_balance
            
            # If we've grown substantially, consider adjusting position sizing 
            if current_balance > self.initial_balance * 1.5:  # 50% growth
                growth_factor = current_balance / self.initial_balance
                # Gradually increase position size as account grows, but not too aggressively
                pos_multiplier = min(1.5, 1.0 + (growth_factor - 1) * 0.5)
                logger.info(f"Account has grown by {(growth_factor-1)*100:.1f}%, adjusting position size multiplier to {pos_multiplier:.2f}")
                self.update_position_sizing(pos_multiplier)
        
        return current_balance

    def calculate_partial_take_profits(self, symbol, side, entry_price):
        """
        Calculate partial take profit levels for tiered profit taking
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price of the position
            
        Returns:
            list: List of take profit levels with quantity percentages
                Each item is a dict with 'price' and 'percentage' keys
        """
        # Define take profit tiers based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            # More optimistic targets in bullish markets
            take_profit_tiers = [
                {'level': TAKE_PROFIT_PCT_BULLISH * 0.5, 'percentage': 0.3},  # 30% of position at first target
                {'level': TAKE_PROFIT_PCT_BULLISH, 'percentage': 0.4},        # 40% of position at second target
                {'level': TAKE_PROFIT_PCT_BULLISH * 2.0, 'percentage': 0.3}   # 30% of position at third target
            ]
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            # More conservative targets in bearish markets
            take_profit_tiers = [
                {'level': TAKE_PROFIT_PCT_BEARISH * 0.7, 'percentage': 0.4},  # 40% of position at first target
                {'level': TAKE_PROFIT_PCT_BEARISH, 'percentage': 0.4},        # 40% of position at second target
                {'level': TAKE_PROFIT_PCT_BEARISH * 1.5, 'percentage': 0.2}   # 20% of position at third target
            ]
        else:  # SIDEWAYS or default
            # Balanced targets in sideways markets
            take_profit_tiers = [
                {'level': TAKE_PROFIT_PCT_SIDEWAYS * 0.6, 'percentage': 0.3},  # 30% of position at first target
                {'level': TAKE_PROFIT_PCT_SIDEWAYS, 'percentage': 0.5},        # 50% of position at second target
                {'level': TAKE_PROFIT_PCT_SIDEWAYS * 1.7, 'percentage': 0.2}   # 20% of position at third target
            ]
            
        # Calculate price levels
        tp_levels = []
        for tier in take_profit_tiers:
            if side == "BUY":  # Long position
                price = entry_price * (1 + tier['level'])
            else:  # Short position
                price = entry_price * (1 - tier['level'])
                
            # Apply price precision
            symbol_info = self.binance_client.get_symbol_info(symbol)
            if symbol_info:
                price_precision = symbol_info['price_precision']
                price = round(price, price_precision)
                
            tp_levels.append({
                'price': price,
                'percentage': tier['percentage']
            })
            
        logger.info(f"Calculated partial take profit levels for {symbol} ({self.current_market_condition} market):")
        for i, level in enumerate(tp_levels):
            logger.info(f"   TP {i+1}: {level['price']} ({level['percentage']*100:.0f}% of position)")
            
        return tp_levels
        
    def calculate_volatility_based_stop_loss(self, symbol, side, entry_price, klines=None):
        """
        Calculate stop loss based on historical volatility (ATR)
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price of the position
            klines: Optional klines data for ATR calculation
            
        Returns:
            float: Stop loss price
        """
        # Default ATR multipliers based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            atr_multiplier = 2.0  # More room in bullish markets
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            atr_multiplier = 1.5  # Tighter stops in bearish markets
        elif self.current_market_condition == 'SIDEWAYS':
            atr_multiplier = 1.8  # Medium stops in sideways markets
        else:
            atr_multiplier = 1.7  # Default
            
        # If we don't have klines data, fetch it
        if klines is None:
            from modules.config import TIMEFRAME
            klines = self.binance_client.get_historical_klines(symbol, TIMEFRAME, "1 day ago")
            
        # If we still don't have data, use the regular percentage-based stop loss
        if not klines or len(klines) < 15:
            logger.warning(f"Insufficient data for volatility-based stop loss. Using regular stop loss instead.")
            return self.calculate_stop_loss(symbol, side, entry_price)
            
        # Calculate ATR
        import pandas as pd
        import ta
        
        # Convert klines to dataframe
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert string values to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        # Calculate ATR with a 14-period window
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
        
        # Get the latest ATR value
        latest_atr = df['atr'].iloc[-1]
        
        # Special handling for high volatility tokens
        is_high_volatility = symbol[-4:] in ["USDT"] and any(token in symbol[:-4] for token in ["SOL", "RAY", "ARB", "DOGE", "SHIB"])
        if is_high_volatility:
            atr_multiplier *= 1.3  # 30% wider for high volatility tokens
            
        # Calculate stop loss price
        if side == "BUY":  # Long position
            stop_loss = entry_price - (latest_atr * atr_multiplier)
        else:  # Short position
            stop_loss = entry_price + (latest_atr * atr_multiplier)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            stop_loss = round(stop_loss, price_precision)
            
        logger.info(f"Calculated volatility-based stop loss at {stop_loss} (ATR: {latest_atr:.4f}, multiplier: {atr_multiplier:.1f})")
        
        # Store the stop loss level
        self.stop_loss_levels[symbol] = stop_loss
        
        return stop_loss
    
    def check_trailing_stops(self, symbol, current_price):
        """
        Check and update trailing stops for an open position
        
        Args:
            symbol: Trading pair symbol
            current_price: Current market price
            
        Returns:
            dict: Updated status with any changes to stops or take profits
        """
        result = {
            'trailing_stop_updated': False,
            'trailing_tp_updated': False
        }
        
        # Get position info
        position_info = self.binance_client.get_position_info(symbol)
        if not position_info or float(position_info['position_amount']) == 0:
            return result  # No open position
            
        # Determine position side
        position_amount = float(position_info['position_amount'])
        side = "BUY" if position_amount > 0 else "SELL"
        
        # Check for trailing stop updates
        if TRAILING_STOP and symbol in self.stop_loss_levels:
            result['trailing_stop_updated'] = self.adjust_stop_loss_for_trailing(
                symbol, side, current_price, position_info
            )
            
        # Check for trailing take profit updates
        if TRAILING_TAKE_PROFIT and symbol in self.take_profit_levels:
            result['trailing_tp_updated'] = self.adjust_take_profit_for_trailing(
                symbol, side, current_price, position_info
            )
            
        if result['trailing_stop_updated'] or result['trailing_tp_updated']:
            logger.info(f"Trailing orders updated for {symbol} at price {current_price}")
            
        return result