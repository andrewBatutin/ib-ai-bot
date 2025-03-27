import threading
import time
import requests
import json
from collections import deque
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('price_monitor')

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
                'last_update': 0
            }
            
            self.price_history[conid] = deque(maxlen=self.history_size)
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
                        
            logger.debug(f"Updated prices for {len(data)} stocks")
                
        except Exception as e:
            logger.error(f"Error fetching prices: {str(e)}")

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