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
            
        # Choose take profit percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            take_profit_pct = TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            take_profit_pct = TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            take_profit_pct = TAKE_PROFIT_PCT_SIDEWAYS
        else:
            take_profit_pct = TAKE_PROFIT_PCT  # Default
            
        if side == "BUY":  # Long position
            take_profit_price = entry_price * (1 + take_profit_pct)
        else:  # Short position
            take_profit_price = entry_price * (1 - take_profit_pct)
            
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            take_profit_price = round(take_profit_price, price_precision)
            
        logger.info(f"Calculated {self.current_market_condition} take profit at {take_profit_price} ({take_profit_pct*100:.2f}%)")
        
        # Store the take profit level for this symbol
        self.take_profit_levels[symbol] = take_profit_price
        
        return take_profit_price
        
    def adjust_stop_loss_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust stop loss for trailing stop if needed
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('BUY' or 'SELL')
            current_price: Current market price
            position_info: Position information dictionary
            
        Returns:
            new_stop: New stop loss price if it should be adjusted, None otherwise
        """
        if not TRAILING_STOP:
            return None
            
        if not position_info:
            # Get position info specifically for this symbol (important for multi-instance mode)
            position_info = self.binance_client.get_position_info(symbol)
            
        # Only proceed if we have a valid position for this specific symbol
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        # Ensure we're dealing with the right symbol in multi-instance mode
        if position_info['symbol'] != symbol:
            logger.warning(f"Position symbol mismatch: expected {symbol}, got {position_info['symbol']}")
            return None
            
        entry_price = position_info['entry_price']
        
        # Choose trailing stop percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            trailing_stop_pct = TRAILING_STOP_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            trailing_stop_pct = TRAILING_STOP_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_stop_pct = TRAILING_STOP_PCT_SIDEWAYS
        else:
            trailing_stop_pct = TRAILING_STOP_PCT  # Default
        
        # Calculate new stop loss based on current price
        if side == "BUY":  # Long position
            new_stop = current_price * (1 - trailing_stop_pct)
            # Only move stop loss up, never down
            current_stop = self.stop_loss_levels.get(symbol) or self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop <= current_stop:
                logger.debug(f"Not adjusting trailing stop for long position: current ({current_stop}) > calculated ({new_stop})")
                return None
        else:  # Short position
            new_stop = current_price * (1 + trailing_stop_pct)
            # Only move stop loss down, never up
            current_stop = self.stop_loss_levels.get(symbol) or self.calculate_stop_loss(symbol, side, entry_price)
            if current_stop and new_stop >= current_stop:
                logger.debug(f"Not adjusting trailing stop for short position: current ({current_stop}) < calculated ({new_stop})")
                return None
                
        # Apply price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if symbol_info:
            price_precision = symbol_info['price_precision']
            new_stop = round(new_stop, price_precision)
        
        # Check if this is a significant move (at least 0.1% change) or if we haven't updated in a while
        last_update_time = self.last_trailing_stop_update.get(symbol, datetime.min)
        significant_move = (abs(new_stop - current_stop) / current_stop > 0.001)
        time_since_update = (datetime.now() - last_update_time).total_seconds()
        
        if significant_move or time_since_update > 300:  # 5 minutes
            logger.info(f"Adjusted {self.current_market_condition} trailing stop loss to {new_stop} ({trailing_stop_pct*100:.2f}%)")
            logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Stop loss moved: {current_stop} -> {new_stop}")
            
            # Update the stop loss level for this symbol
            self.stop_loss_levels[symbol] = new_stop
            self.last_trailing_stop_update[symbol] = datetime.now()
            
            return new_stop
        else:
            logger.debug(f"Skipping minor trailing stop update (last update: {time_since_update:.1f}s ago)")
            return None
        
    def adjust_take_profit_for_trailing(self, symbol, side, current_price, position_info=None):
        """
        Adjust take profit price based on trailing settings
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('BUY' or 'SELL')
            current_price: Current market price
            position_info: Position information including entry_price
            
        Returns:
            new_take_profit: New take profit price if it should be adjusted, None otherwise
        """
        if not USE_TAKE_PROFIT or not TRAILING_TAKE_PROFIT:
            return None
            
        if not position_info:
            position_info = self.binance_client.get_position_info(symbol)
            
        if not position_info or abs(position_info['position_amount']) == 0:
            return None
            
        entry_price = float(position_info.get('entry_price', 0))
        if entry_price <= 0:
            return None
        
        # Get symbol info for precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        if not symbol_info:
            return None
            
        price_precision = symbol_info.get('price_precision', 2)
        
        # Choose trailing take profit percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BULLISH
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_BEARISH
        elif self.current_market_condition == 'SIDEWAYS':
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT_SIDEWAYS
        else:
            trailing_take_profit_pct = TRAILING_TAKE_PROFIT_PCT  # Default
        
        # Calculate the current dynamic take profit level based on the current price
        if side == 'BUY':  # Long position
            # For long positions, we want take profit to trail above the price
            current_take_profit = current_price * (1 + trailing_take_profit_pct)
            current_take_profit = math.floor(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders specifically for this symbol
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists - only for this specific symbol
            # This is crucial for multi-instance mode to prevent conflicts between different trading pairs
            existing_take_profit = None
            for order in open_orders:
                if (order['symbol'] == symbol and 
                    order['type'] == 'TAKE_PROFIT_MARKET' and 
                    order['side'] == 'SELL'):
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit, use the stored value or calculate a new one
            if not existing_take_profit:
                existing_take_profit = self.take_profit_levels.get(symbol) or self.calculate_take_profit(symbol, side, entry_price)
            
            # If no existing take profit or our new one is better (higher for long), return the new one
            if not existing_take_profit:
                logger.info(f"Long position: Setting initial {self.current_market_condition} take profit to {current_take_profit} ({trailing_take_profit_pct*100:.2f}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}")
                
                # Update the take profit level for this symbol
                self.take_profit_levels[symbol] = current_take_profit
                self.last_trailing_tp_update[symbol] = datetime.now()
                
                return current_take_profit
            elif current_take_profit > existing_take_profit:
                # Check if this is a significant move or if we haven't updated in a while
                last_update_time = self.last_trailing_tp_update.get(symbol, datetime.min)
                significant_move = (abs(current_take_profit - existing_take_profit) / existing_take_profit > 0.001)
                time_since_update = (datetime.now() - last_update_time).total_seconds()
                
                if significant_move or time_since_update > 300:  # 5 minutes
                    logger.info(f"Long position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100:.2f}%)")
                    logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {existing_take_profit} -> {current_take_profit}")
                    
                    # Update the take profit level for this symbol
                    self.take_profit_levels[symbol] = current_take_profit
                    self.last_trailing_tp_update[symbol] = datetime.now()
                    
                    return current_take_profit
                else:
                    logger.debug(f"Skipping minor trailing take profit update (last update: {time_since_update:.1f}s ago)")
                    return None
            else:
                logger.debug(f"Not adjusting trailing take profit for long position: current ({existing_take_profit}) > calculated ({current_take_profit})")
                return None
                
        elif side == 'SELL':  # Short position
            # For short positions, we want take profit to trail below the price
            current_take_profit = current_price * (1 - trailing_take_profit_pct)
            current_take_profit = math.ceil(current_take_profit * 10**price_precision) / 10**price_precision
            
            # Check if there are open orders specifically for this symbol
            open_orders = self.binance_client.client.futures_get_open_orders(symbol=symbol)
            
            # Find the current take profit order if it exists - only for this specific symbol
            # This is crucial for multi-instance mode to prevent conflicts between different trading pairs
            existing_take_profit = None
            for order in open_orders:
                if (order['symbol'] == symbol and
                    order['type'] == 'TAKE_PROFIT_MARKET' and 
                    order['side'] == 'BUY'):
                    existing_take_profit = float(order['stopPrice'])
                    break
            
            # If no existing take profit, use the stored value or calculate a new one
            if not existing_take_profit:
                existing_take_profit = self.take_profit_levels.get(symbol) or self.calculate_take_profit(symbol, side, entry_price)
            
            # If no existing take profit or our new one is better (lower), return the new one
            if not existing_take_profit:
                logger.info(f"Short position: Setting initial {self.current_market_condition} take profit to {current_take_profit} ({trailing_take_profit_pct*100:.2f}%)")
                logger.info(f"Current price: {current_price}, Entry price: {entry_price}")
                
                # Update the take profit level for this symbol
                self.take_profit_levels[symbol] = current_take_profit
                self.last_trailing_tp_update[symbol] = datetime.now()
                
                return current_take_profit
            elif current_take_profit < existing_take_profit:
                # Check if this is a significant move or if we haven't updated in a while
                last_update_time = self.last_trailing_tp_update.get(symbol, datetime.min)
                significant_move = (abs(current_take_profit - existing_take_profit) / existing_take_profit > 0.001)
                time_since_update = (datetime.now() - last_update_time).total_seconds()
                
                if significant_move or time_since_update > 300:  # 5 minutes
                    logger.info(f"Short position: Adjusting {self.current_market_condition} take profit from {existing_take_profit} to {current_take_profit} ({trailing_take_profit_pct*100:.2f}%)")
                    logger.info(f"Current price: {current_price}, Entry price: {entry_price}, Take profit moved: {existing_take_profit} -> {current_take_profit}")
                    
                    # Update the take profit level for this symbol
                    self.take_profit_levels[symbol] = current_take_profit
                    self.last_trailing_tp_update[symbol] = datetime.now()
                    
                    return current_take_profit
                else:
                    logger.debug(f"Skipping minor trailing take profit update (last update: {time_since_update:.1f}s ago)")
                    return None
            else:
                logger.debug(f"Not adjusting trailing take profit for short position: current ({existing_take_profit}) < calculated ({current_take_profit})")
                return None
        
        return None
        
    def update_balance_for_compounding(self):
        """Update balance tracking for auto-compounding"""
        if not AUTO_COMPOUND:
            return False
            
        current_balance = self.binance_client.get_account_balance()
        
        # First time initialization
        if self.last_known_balance is None:
            self.last_known_balance = current_balance
            self.initial_balance = current_balance
            return False
        
        profit = current_balance - self.last_known_balance
        
        if profit > 0:
            # We've made profits since last update
            reinvest_amount = profit * COMPOUND_REINVEST_PERCENT
            logger.info(f"Auto-compounding: {reinvest_amount:.2f} USDT from recent {profit:.2f} USDT profit")
            
            # Update balance after compounding
            self.last_known_balance = current_balance
            return True
            
        return False

    def calculate_partial_take_profits(self, symbol, side, entry_price):
        """
        Calculate multiple partial take profit levels based on market condition
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price of the position
            
        Returns:
            list: List of dictionaries with take profit levels and percentages of position to close
        """
        if not USE_TAKE_PROFIT:
            return []
            
        # Choose take profit percentage based on market condition
        if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
            tp1_pct = TAKE_PROFIT_PCT_BULLISH * 0.5  # 50% of target
            tp2_pct = TAKE_PROFIT_PCT_BULLISH        # 100% of target
            tp3_pct = TAKE_PROFIT_PCT_BULLISH * 1.5  # 150% of target
        elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
            tp1_pct = TAKE_PROFIT_PCT_BEARISH * 0.5  # Earlier take profit in bearish market
            tp2_pct = TAKE_PROFIT_PCT_BEARISH
            tp3_pct = TAKE_PROFIT_PCT_BEARISH * 1.3  # Only 130% of target in bearish markets
        elif self.current_market_condition == 'SIDEWAYS':
            tp1_pct = TAKE_PROFIT_PCT_SIDEWAYS * 0.7  # Take profits quicker in sideways markets
            tp2_pct = TAKE_PROFIT_PCT_SIDEWAYS
            tp3_pct = TAKE_PROFIT_PCT_SIDEWAYS * 1.2  # Conservative extension in sideways
        else:
            tp1_pct = TAKE_PROFIT_PCT * 0.5
            tp2_pct = TAKE_PROFIT_PCT
            tp3_pct = TAKE_PROFIT_PCT * 1.5
        
        # Get symbol info for price precision
        symbol_info = self.binance_client.get_symbol_info(symbol)
        price_precision = 2  # Default
        if symbol_info:
            price_precision = symbol_info.get('price_precision', 2)
        
        # Calculate take profit prices
        if side == "BUY":  # Long position
            tp1_price = round(entry_price * (1 + tp1_pct), price_precision)
            tp2_price = round(entry_price * (1 + tp2_pct), price_precision)
            tp3_price = round(entry_price * (1 + tp3_pct), price_precision)
        else:  # Short position
            tp1_price = round(entry_price * (1 - tp1_pct), price_precision)
            tp2_price = round(entry_price * (1 - tp2_pct), price_precision)
            tp3_price = round(entry_price * (1 - tp3_pct), price_precision)
        
        # Define partial take profit levels with % of position to close at each level
        take_profits = [
            {'price': tp1_price, 'percentage': 0.3, 'pct_from_entry': tp1_pct * 100},  # Close 30% at first TP
            {'price': tp2_price, 'percentage': 0.4, 'pct_from_entry': tp2_pct * 100},  # Close 40% at second TP
            {'price': tp3_price, 'percentage': 0.3, 'pct_from_entry': tp3_pct * 100}   # Close 30% at third TP
        ]
        
        logger.info(f"Calculated {self.current_market_condition} partial take profits: "
                   f"TP1: {tp1_price} ({tp1_pct*100:.2f}%), "
                   f"TP2: {tp2_price} ({tp2_pct*100:.2f}%), "
                   f"TP3: {tp3_price} ({tp3_pct*100:.2f}%)")
                   
        return take_profits
        
    def calculate_volatility_based_stop_loss(self, symbol, side, entry_price, klines=None):
        """
        Calculate stop loss based on volatility (ATR) rather than fixed percentage
        
        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            entry_price: Entry price
            klines: Optional recent price data for ATR calculation
            
        Returns:
            float: Volatility-adjusted stop loss price
        """
        if not USE_STOP_LOSS:
            return None
            
        # If no klines provided, use default percentage-based stop loss
        if klines is None or len(klines) < 14:
            return self.calculate_stop_loss(symbol, side, entry_price)
        
        # Special handling for high volatility tokens
        is_high_volatility = symbol[-4:] in ["USDT"] and any(token in symbol[:-4] for token in ["SOL", "RAY", "ARB", "DOGE", "SHIB"])
            
        try:
            # Convert klines to dataframe for ATR calculation
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert string values to numeric
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col])
                
            # Calculate ATR
            atr_period = 14
            if len(df) >= atr_period:
                # Use ta library ATR
                atr = ta.volatility.average_true_range(
                    df['high'], df['low'], df['close'], window=atr_period
                ).iloc[-1]
                
                # Calculate ATR as percentage of price
                atr_pct = atr / entry_price
                
                # Base multiplier on market condition
                if self.current_market_condition in ['BULLISH', 'EXTREME_BULLISH']:
                    atr_multiplier = 2.0  # Wider stops in bullish trend
                elif self.current_market_condition in ['BEARISH', 'EXTREME_BEARISH']:
                    atr_multiplier = 1.5  # Medium stops in bearish trend
                else:  # SIDEWAYS or SQUEEZE
                    atr_multiplier = 1.0  # Tighter stops in sideways market
                
                # Apply high volatility token specific adjustments - increase multiplier by 50%
                if is_high_volatility:
                    original_multiplier = atr_multiplier
                    atr_multiplier = atr_multiplier * 1.5
                    logger.info(f"High volatility token detected: Increasing ATR multiplier from {original_multiplier} to {atr_multiplier}")
                
                # Calculate stop loss price - use ATR * multiplier but cap it
                if side == "BUY":  # Long
                    # Cap maximum stop distance to standard percentage stop loss
                    max_stop_distance = entry_price * STOP_LOSS_PCT * 1.5  # Allow 50% more than standard
                    # For high volatility tokens, increase the maximum stop distance by 50%
                    if is_high_volatility:
                        max_stop_distance = max_stop_distance * 1.5
                    atr_stop_distance = min(atr * atr_multiplier, max_stop_distance)
                    stop_price = entry_price - atr_stop_distance
                else:  # Short
                    max_stop_distance = entry_price * STOP_LOSS_PCT * 1.5  # Allow 50% more than standard
                    # For high volatility tokens, increase the maximum stop distance by 50%
                    if is_high_volatility:
                        max_stop_distance = max_stop_distance * 1.5
                    atr_stop_distance = min(atr * atr_multiplier, max_stop_distance)
                    stop_price = entry_price + atr_stop_distance
                
                # Apply price precision
                symbol_info = self.binance_client.get_symbol_info(symbol)
                if symbol_info:
                    price_precision = symbol_info['price_precision']
                    stop_price = round(stop_price, price_precision)
                    
                # Add high volatility token specific buffer information to log
                if is_high_volatility:
                    logger.info(f"Calculated high volatility token ATR-based stop loss at {stop_price} "
                              f"(ATR: {atr:.6f}, {atr_pct*100:.2f}% of price, "
                              f"Multiplier: {atr_multiplier}, Enhanced buffer active)")
                else:
                    logger.info(f"Calculated ATR-based stop loss at {stop_price} "
                              f"(ATR: {atr:.6f}, {atr_pct*100:.2f}% of price, "
                              f"Multiplier: {atr_multiplier})")
                
                # Store this stop loss level
                self.stop_loss_levels[symbol] = stop_price
                
                return stop_price
                
        except Exception as e:
            logger.error(f"Error calculating volatility-based stop loss: {e}")
            
        # Fall back to standard stop loss if ATR calculation fails
        return self.calculate_stop_loss(symbol, side, entry_price)
        
    def check_trailing_stops(self, symbol, current_price):
        """
        Check and adjust trailing stop and take profit orders for a symbol
        
        Args:
            symbol: Trading pair symbol
            current_price: Current price of the asset
            
        Returns:
            dict: Dictionary with updated stop loss and take profit prices
        """
        updates = {
            'stop_loss_updated': False,
            'take_profit_updated': False,
            'new_stop_loss': None,
            'new_take_profit': None
        }
        
        # Get position info
        position_info = self.binance_client.get_position_info(symbol)
        if not position_info or abs(position_info['position_amount']) == 0:
            return updates
        
        # Determine position side
        position_amount = position_info['position_amount']
        side = "BUY" if position_amount > 0 else "SELL"
        
        # Check if trailing stop should be updated
        if TRAILING_STOP:
            new_stop = self.adjust_stop_loss_for_trailing(symbol, side, current_price, position_info)
            if new_stop:
                updates['stop_loss_updated'] = True
                updates['new_stop_loss'] = new_stop
        
        # Check if trailing take profit should be updated
        if TRAILING_TAKE_PROFIT:
            new_tp = self.adjust_take_profit_for_trailing(symbol, side, current_price, position_info)
            if new_tp:
                updates['take_profit_updated'] = True
                updates['new_take_profit'] = new_tp
                
        return updates


def round_step_size(quantity, step_size):
    """Round quantity based on step size"""
    precision = int(round(-math.log10(step_size)))
    return round(math.floor(quantity * 10**precision) / 10**precision, precision)


def get_step_size(min_qty):
    """Get step size from min_qty"""
    step_size = min_qty
    # Handle cases where min_qty is not the step size (common in Binance)
    if float(min_qty) > 0:
        step_size = float(min_qty)
        
    return step_size