import threading
import time
import requests
import json
from collections import deque
import logging
from datetime import datetime, date, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('price_monitor')

class QStocks:
    """Class to handle Q-signal and normalized price calculations"""
    def __init__(self):
        # Q-stocks tickers from the notebook
        self.tickers = ['IONQ', 'RGTI', 'QMCO', 'QUBT', 'QBTS']
        self.q_stocks_conids = {}  # symbol -> conid mapping
        self.reference_prices = {}  # conid -> first price for normalization
        self.q_signal_history = deque(maxlen=100)  # Store Q-signal history
        
    def add_conid(self, symbol, conid):
        """Map a symbol to its conid for Q-stocks tracking"""
        if symbol.upper() in self.tickers:
            self.q_stocks_conids[symbol.upper()] = conid
            return True
        return False
        
    def is_q_stock(self, symbol):
        """Check if a symbol is in the Q-stocks list"""
        return symbol.upper() in self.tickers
        
    def has_all_tickers(self):
        """Check if all required Q-stocks tickers are tracked"""
        return all(ticker in self.q_stocks_conids for ticker in self.tickers)
        
    def calculate_q_signal(self, price_data):
        """Calculate normalized prices and Q-signal from current price data"""
        if not self.has_all_tickers() or not price_data:
            return None
        
        # Get latest prices for all Q-stocks
        latest_prices = {}
        timestamp = int(time.time() * 1000)
        
        for ticker in self.tickers:
            conid = self.q_stocks_conids.get(ticker)
            if not conid or conid not in price_data:
                return None
                
            # Get latest price
            stock_data = price_data.get(conid)
            if not stock_data or 'latest' not in stock_data or not stock_data['latest']:
                return None
                
            price = stock_data['latest']['price']
            latest_prices[ticker] = price
            
            # Store reference price (first price seen) for normalization
            if conid not in self.reference_prices:
                self.reference_prices[conid] = price
        
        # Calculate normalized prices
        normalized_prices = {}
        for ticker in self.tickers:
            conid = self.q_stocks_conids.get(ticker)
            ref_price = self.reference_prices.get(conid, 1.0)  # Default to 1 if no reference
            current_price = latest_prices.get(ticker, 0.0)
            normalized_prices[ticker] = current_price / ref_price if ref_price else 0.0
        
        # Calculate Q-signal (average of normalized prices)
        q_signal = sum(normalized_prices.values()) / len(normalized_prices) if normalized_prices else 0.0
        
        # Store in history
        self.q_signal_history.append({
            'timestamp': timestamp,
            'q_signal': q_signal,
            'normalized_prices': normalized_prices
        })
        
        return {
            'timestamp': timestamp,
            'q_signal': q_signal,
            'normalized_prices': normalized_prices
        }
    
    def get_q_signal_history(self):
        """Get the full history of Q-signals and normalized prices"""
        return list(self.q_signal_history)


class PriceMonitor:
    def __init__(self, base_api_url, max_stocks=10, history_size=100):
        self.base_api_url = base_api_url
        self.max_stocks = max_stocks
        self.history_size = history_size
        self.monitored_stocks = {}  # conid -> {symbol, name, ...}
        self.price_history = {}     # conid -> deque of {timestamp, price} objects
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.q_stocks = QStocks()  # Initialize Q-stocks calculator
        
    def start(self):
        """Start the price monitoring thread"""
        if self.running:
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info("Price monitor started")
        return True
        
    def stop(self):
        """Stop the price monitoring thread"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info("Price monitor stopped")
        
    def add_stock(self, conid, symbol, name):
        """Add a stock to the monitoring list"""
        with self.lock:
            if len(self.monitored_stocks) >= self.max_stocks:
                return False, "Maximum number of monitored stocks reached"
                
            if conid in self.monitored_stocks:
                return False, "Stock already being monitored"
                
            self.monitored_stocks[conid] = {
                'conid': conid,
                'symbol': symbol,
                'name': name,
                'last_update': 0,
                'is_q_stock': self.q_stocks.is_q_stock(symbol)
            }
            
            self.price_history[conid] = deque(maxlen=self.history_size)
            
            # If this is a Q-stock, add it to the Q-stocks tracker
            if self.q_stocks.is_q_stock(symbol):
                self.q_stocks.add_conid(symbol, conid)
                logger.info(f"Added Q-stock to monitor: {symbol} ({conid})")
            
            logger.info(f"Added stock to monitor: {symbol} ({conid})")
            return True, "Stock added successfully"
            
    def remove_stock(self, conid):
        """Remove a stock from the monitoring list"""
        with self.lock:
            if conid in self.monitored_stocks:
                symbol = self.monitored_stocks[conid]['symbol']
                del self.monitored_stocks[conid]
                if conid in self.price_history:
                    del self.price_history[conid]
                logger.info(f"Removed stock from monitor: {symbol} ({conid})")
                return True, "Stock removed successfully"
            return False, "Stock not found"
            
    def get_monitored_stocks(self):
        """Get list of currently monitored stocks"""
        with self.lock:
            return list(self.monitored_stocks.values())
            
    def get_stock_data(self, conid):
        """Get current data and price history for a specific stock"""
        with self.lock:
            if conid not in self.monitored_stocks:
                return None
                
            return {
                'stock': self.monitored_stocks[conid],
                'history': list(self.price_history[conid])
            }
            
    def get_latest_prices(self):
        """Get the latest price for all monitored stocks"""
        with self.lock:
            result = {}
            for conid, stock in self.monitored_stocks.items():
                history = self.price_history.get(conid, deque())
                latest = history[-1] if history else None
                result[conid] = {
                    'stock': stock,
                    'latest': latest
                }
            return result
            
    def _monitor_loop(self):
        """Main monitoring loop that runs in a background thread"""
        while self.running:
            try:
                self._update_prices()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in price monitor: {str(e)}")
                time.sleep(10)  # Wait a bit longer after an error
                
    def _update_prices(self):
        """Request updated prices from IB API for all monitored stocks"""
        with self.lock:
            conids = list(self.monitored_stocks.keys())
            
        if not conids:
            return  # No stocks to monitor
            
        # Format conids for the API request
        conid_param = ",".join(map(str, conids))
        
        try:
            # Request snapshot data for all monitored stocks
            url = f"{self.base_api_url}/iserver/marketdata/snapshot?conids={conid_param}&fields=31,84,86"
            response = requests.get(url, verify=False)
            
            if response.status_code != 200:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return
                
            data = response.json()
            
            with self.lock:
                timestamp = int(time.time() * 1000)  # Current time in milliseconds
                
                for item in data:
                    conid = str(item.get('conid'))
                    if conid not in self.monitored_stocks:
                        continue
                        
                    # Extract price - field 31 is last price
                    if '31' not in item:
                        continue
                        
                    price = float(item['31'])
                    
                    # Update stock data
                    self.monitored_stocks[conid]['last_price'] = price
                    self.monitored_stocks[conid]['last_update'] = timestamp
                    
                    # Add to price history
                    self.price_history[conid].append({
                        'timestamp': timestamp,
                        'price': price
                    })
                    
                    # Additional fields if available
                    if '84' in item:  # bid price
                        self.monitored_stocks[conid]['bid'] = float(item['84'])
                    if '86' in item:  # ask price
                        self.monitored_stocks[conid]['ask'] = float(item['86'])
                
                # Calculate Q-signal if possible
                latest_prices = self.get_latest_prices()
                q_data = self.q_stocks.calculate_q_signal(latest_prices)
                if q_data:
                    logger.debug(f"Updated Q-signal: {q_data['q_signal']}")
                        
            logger.debug(f"Updated prices for {len(data)} stocks")
                
        except Exception as e:
            logger.error(f"Error fetching prices: {str(e)}")

    def get_q_signal_data(self):
        """Get Q-signal data and status"""
        with self.lock:
            # Get the list of Q-stocks tickers
            q_tickers = self.q_stocks.tickers
            
            # Check which Q-stocks are being monitored
            monitored_q_stocks = {}
            for conid, stock in self.monitored_stocks.items():
                if stock.get('is_q_stock', False):
                    monitored_q_stocks[stock['symbol']] = {
                        'conid': conid,
                        'name': stock['name'],
                        'monitored': True
                    }
            
            # Add missing Q-stocks
            missing_q_stocks = []
            for ticker in q_tickers:
                if ticker not in monitored_q_stocks:
                    missing_q_stocks.append(ticker)
                    monitored_q_stocks[ticker] = {
                        'monitored': False
                    }
            
            # Get Q-signal history
            q_signal_history = self.q_stocks.get_q_signal_history()
            
            has_all_tickers = self.q_stocks.has_all_tickers()
            
            return {
                'q_stocks': monitored_q_stocks,
                'missing_tickers': missing_q_stocks,
                'has_all_tickers': has_all_tickers,
                'q_signal_history': q_signal_history
            }

    def get_required_q_stocks(self):
        """Get the list of required Q-stocks tickers"""
        return self.q_stocks.tickers

# Singleton instance
_instance = None

def init_price_monitor(base_api_url, max_stocks=10, history_size=100):
    """Initialize the price monitor singleton"""
    global _instance
    if _instance is None:
        _instance = PriceMonitor(base_api_url, max_stocks, history_size)
    return _instance
    
def get_price_monitor():
    """Get the price monitor singleton instance"""
    global _instance
    if _instance is None:
        raise RuntimeError("Price monitor not initialized")
    return _instance