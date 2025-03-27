import asyncio
import time
import aiohttp
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
        self.task = None
        self.session = None
        self.q_stocks = QStocks()  # Initialize Q-stocks calculator
        
    async def start(self):
        """Start the price monitoring task"""
        if self.running:
            return False
            
        self.running = True
        self.session = aiohttp.ClientSession()
        self.task = asyncio.create_task(self._monitor_loop())
        logger.info("Price monitor started")
        return True
        
    async def stop(self):
        """Stop the price monitoring task"""
        self.running = False
        if self.task:
            try:
                self.task.cancel()
                await self.task
            except asyncio.CancelledError:
                pass
            
        if self.session:
            await self.session.close()
            self.session = None
            
        logger.info("Price monitor stopped")
        
    def add_stock(self, conid, symbol, name):
        """Add a stock to the monitoring list"""
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
        return list(self.monitored_stocks.values())
        
    def get_stock_data(self, conid):
        """Get current data and price history for a specific stock"""
        if conid not in self.monitored_stocks:
            logger.warning(f"Stock not found in monitored_stocks: {conid}")
            return None
            
        # Use deep copy to avoid reference issues
        stock_copy = {k: v for k, v in self.monitored_stocks[conid].items()}
        
        # Get history and ensure it's not empty
        history = list(self.price_history.get(conid, []))
        
        # If no history, create a dummy entry with current timestamp and last_price
        if not history and 'last_price' in stock_copy:
            current_time = int(time.time() * 1000)
            dummy_entry = {
                'timestamp': current_time,
                'price': stock_copy.get('last_price', 0.0)
            }
            history.append(dummy_entry)
            logger.info(f"Created dummy history for {stock_copy.get('symbol')} with price {dummy_entry['price']}")
        
        logger.info(f"Returning {len(history)} history points for {stock_copy.get('symbol')}")
        
        return {
            'stock': stock_copy,
            'history': history
        }
        
    def get_latest_prices(self):
        """Get the latest price for all monitored stocks"""
        result = {}
        for conid, stock in self.monitored_stocks.items():
            history = self.price_history.get(conid, deque())
            latest = history[-1] if history and len(history) > 0 else None
            
            # Create a deep copy of stock data to avoid reference issues
            stock_copy = {k: v for k, v in stock.items()}
            
            # Ensure all required fields exist
            if 'last_price' not in stock_copy:
                stock_copy['last_price'] = 0.0
            if 'last_update' not in stock_copy:
                stock_copy['last_update'] = int(time.time() * 1000)
                
            result[conid] = {
                'stock': stock_copy,
                'latest': latest
            }
            
            # Debug log for troubleshooting
            if latest:
                logger.debug(f"Latest price for {stock_copy.get('symbol')}: {latest.get('price')}")
            else:
                logger.warning(f"No price history for {stock_copy.get('symbol')} ({conid})")
                
        return result
        
    async def _monitor_loop(self):
        """Main monitoring loop that runs as an async task"""
        while self.running:
            try:
                await self._update_prices()
                await asyncio.sleep(5)  # Update every 5 seconds
            except Exception as e:
                logger.error(f"Error in price monitor: {str(e)}")
                await asyncio.sleep(10)  # Wait a bit longer after an error
                
    async def _update_prices(self):
        """Request updated prices from IB API for all monitored stocks"""
        conids = list(self.monitored_stocks.keys())
            
        if not conids:
            logger.info("No stocks to monitor")
            return  # No stocks to monitor
            
        # Format conids for the API request
        conid_param = ",".join(map(str, conids))
        logger.info(f"Updating prices for {len(conids)} stocks: {conid_param}")
        
        try:
            # Request snapshot data for all monitored stocks
            url = f"{self.base_api_url}/iserver/marketdata/snapshot?conids={conid_param}&fields=31,84,86"
            logger.debug(f"Requesting data from: {url}")
            
            async with self.session.get(url, ssl=False) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"API error: {response.status} - {error_text}")
                    
                    # Even on API error, add a fallback price entry to each stock's history
                    # This ensures we at least have some data to display
                    timestamp = int(time.time() * 1000)  # Current time in milliseconds
                    for conid, stock in self.monitored_stocks.items():
                        if 'last_price' in stock:
                            # Reuse the last known price
                            last_price = stock['last_price']
                        else:
                            # Use a placeholder price of 0.01 if we have none
                            last_price = 0.01
                            stock['last_price'] = last_price
                            
                        # Add dummy entry to history
                        self.price_history.setdefault(conid, deque(maxlen=self.history_size))
                        self.price_history[conid].append({
                            'timestamp': timestamp,
                            'price': last_price
                        })
                        logger.warning(f"Added fallback price for {stock.get('symbol')}: {last_price}")
                    
                    return
                    
                data = await response.json()
                logger.info(f"Received data for {len(data)} stocks")
                
                timestamp = int(time.time() * 1000)  # Current time in milliseconds
                updated_conids = []
                
                for item in data:
                    conid = str(item.get('conid', ''))
                    if not conid or conid not in self.monitored_stocks:
                        logger.warning(f"Received data for unknown conid: {conid}")
                        continue
                        
                    # Extract price - field 31 is last price
                    if '31' not in item:
                        logger.warning(f"No price data for {conid} - Adding fallback price")
                        # Add a fallback price based on last known price
                        if 'last_price' in self.monitored_stocks[conid]:
                            last_price = self.monitored_stocks[conid]['last_price']
                        else:
                            last_price = 0.01
                            self.monitored_stocks[conid]['last_price'] = last_price
                            
                        # Add to price history
                        self.price_history.setdefault(conid, deque(maxlen=self.history_size))
                        self.price_history[conid].append({
                            'timestamp': timestamp,
                            'price': last_price
                        })
                        self.monitored_stocks[conid]['last_update'] = timestamp
                        updated_conids.append(conid)
                        continue
                        
                    try:
                        price = float(item['31'])
                        
                        # Update stock data
                        self.monitored_stocks[conid]['last_price'] = price
                        self.monitored_stocks[conid]['last_update'] = timestamp
                        
                        # Ensure the deque exists
                        self.price_history.setdefault(conid, deque(maxlen=self.history_size))
                        
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
                            
                        updated_conids.append(conid)
                        logger.debug(f"Updated price for {self.monitored_stocks[conid]['symbol']}: {price}")
                    except (ValueError, KeyError) as e:
                        logger.error(f"Error processing data for {conid}: {str(e)}")
                        # Add fallback price in case of error
                        if 'last_price' in self.monitored_stocks[conid]:
                            last_price = self.monitored_stocks[conid]['last_price']
                        else:
                            last_price = 0.01
                            self.monitored_stocks[conid]['last_price'] = last_price
                            
                        # Add to price history
                        self.price_history.setdefault(conid, deque(maxlen=self.history_size))
                        self.price_history[conid].append({
                            'timestamp': timestamp,
                            'price': last_price
                        })
                        self.monitored_stocks[conid]['last_update'] = timestamp
                        updated_conids.append(conid)
                
                # Check if any conids weren't updated
                missing_updates = set(conids) - set(updated_conids)
                if missing_updates:
                    logger.warning(f"Missing price updates for: {', '.join(missing_updates)}")
                    
                    # Add fallback entries for stocks with missing updates
                    for conid in missing_updates:
                        stock = self.monitored_stocks[conid]
                        if 'last_price' in stock:
                            last_price = stock['last_price']
                        else:
                            last_price = 0.01
                            stock['last_price'] = last_price
                            
                        # Add to price history
                        self.price_history.setdefault(conid, deque(maxlen=self.history_size))
                        self.price_history[conid].append({
                            'timestamp': timestamp,
                            'price': last_price
                        })
                        stock['last_update'] = timestamp
                        logger.warning(f"Added fallback price for missing update {stock.get('symbol')}: {last_price}")
                
                # Calculate Q-signal if possible
                latest_prices = self.get_latest_prices()
                q_data = self.q_stocks.calculate_q_signal(latest_prices)
                if q_data:
                    logger.info(f"Updated Q-signal: {q_data['q_signal']}")
                else:
                    logger.warning("Failed to calculate Q-signal")
                        
            logger.info(f"Successfully updated prices for {len(updated_conids)} out of {len(conids)} stocks")
                
        except Exception as e:
            logger.error(f"Error fetching prices: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def get_q_signal_data(self):
        """Get Q-signal data and status"""
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