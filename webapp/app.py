import requests, time, os, random, json, asyncio
from flask import Flask, render_template, request, redirect, jsonify
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date, timedelta
import sys
import logging

# Add fincalc to path (assuming 'webapp' and 'fincalc' are in the same parent directory)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Adjusted import based on the new location of fincalc
from fincalc.portfolio_analytics import calculate_portfolio_beta, dump_portfolio_to_json

# disable warnings until you install a certificate
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

BASE_API_URL = "https://localhost:5055/v1/api"
ACCOUNT_ID = os.environ['IBKR_ACCOUNT_ID']

os.environ['PYTHONHTTPSVERIFY'] = '0'

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

# Import and initialize price monitor
from price_monitor import init_price_monitor, get_price_monitor

# Initialize price monitor
price_monitor = init_price_monitor(BASE_API_URL, max_stocks=10, history_size=100)

# This function will be triggered on shutdown
@app.teardown_appcontext
def shutdown_price_monitor(exception=None):
    if hasattr(app, 'price_monitor_running') and app.price_monitor_running:
        # Can't use async function directly in Flask without async extra
        # We'll handle shutdown separately
        app.price_monitor_running = False

# Initialize monitoring in a thread to not block app startup
def initialize_monitoring():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(price_monitor.start())
    app.price_monitor_running = True
    
# Start monitoring in a separate thread 
import threading
monitoring_thread = threading.Thread(target=initialize_monitoring)
monitoring_thread.daemon = True
monitoring_thread.start()

@app.template_filter('ctime')
def timectime(s):
    return time.ctime(s/1000)


@app.route("/")
def dashboard():
    portfolio_beta = None # Default value
    summary = {}
    account_display = {}

    try:
        r_accounts = requests.get(f"{BASE_API_URL}/portfolio/accounts", verify=False)
        r_accounts.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        accounts = r_accounts.json()
        if not accounts:
             app.logger.warning("No accounts found via API.")
             # Try using the environment variable as a fallback ID source
             if ACCOUNT_ID:
                 accounts = [{'id': ACCOUNT_ID}] # Synthesize account list
             else:
                 return "No accounts found and no IBKR_ACCOUNT_ID set. Please check IBKR connection and account setup.", 500

    except requests.exceptions.RequestException as e:
        app.logger.error(f"Error fetching accounts: {e}")
        return 'Error connecting to IBKR API to fetch accounts. Make sure the gateway is running and authenticated. <a href="https://localhost:5055">Log in</a>', 503
    except Exception as e:
         app.logger.error(f"Unexpected error fetching accounts: {e}")
         return 'An unexpected error occurred while fetching accounts. Check logs.', 500

    # Use the first account
    account = accounts[0]
    account_id = account.get('id')
    account_display = account # Store for passing to template

    if not account_id:
        app.logger.error("Could not determine Account ID from fetched accounts.")
        return "Could not determine Account ID.", 500

    # --- Attempt to dump portfolio to JSON --- 
    try:
        dump_success = dump_portfolio_to_json(BASE_API_URL, account_id)
        if dump_success:
            app.logger.info(f"Successfully dumped portfolio for account {account_id} on dashboard load.")
        else:
            app.logger.warning(f"Failed to dump portfolio for account {account_id} on dashboard load. Check fincalc logs.")
    except Exception as dump_ex:
        app.logger.error(f"Unexpected error calling dump_portfolio_to_json: {dump_ex}", exc_info=True)
        # Continue dashboard load even if dump fails

    try:
        # Fetch Summary
        r_summary = requests.get(f"{BASE_API_URL}/portfolio/{account_id}/summary", verify=False)
        r_summary.raise_for_status()
        summary = r_summary.json()

        # Fetch Positions
        r_positions = requests.get(f"{BASE_API_URL}/portfolio/{account_id}/positions/0", verify=False)
        r_positions.raise_for_status()
        positions_raw = r_positions.json()

        # --- Beta Calculation --- 
        if positions_raw and isinstance(positions_raw, list):
            holdings_list = []
            symbols = set() # Use set for unique symbols
            for pos in positions_raw:
                # Ensure pos is a dictionary and necessary keys exist
                if isinstance(pos, dict) and 'contractDesc' in pos and 'mktValue' in pos and pos.get('mktValue') is not None:
                    # Try 'ticker' first, then extract from 'contractDesc'
                    symbol = pos.get('ticker')
                    if not symbol:
                         # Basic extraction, might need refinement for complex contracts
                         parts = pos.get('contractDesc', '').split(' ')
                         if parts:
                             symbol = parts[0]

                    market_value = pos.get('mktValue')

                    # Validate symbol and value before adding
                    if symbol and isinstance(market_value, (int, float)):
                        symbols.add(symbol)
                        holdings_list.append({
                            'Symbol': symbol,
                            'Value': float(market_value) # Ensure value is float
                        })
                    else:
                         app.logger.debug(f"Skipping position due to missing symbol or invalid market value: {pos}")
                else:
                     app.logger.debug(f"Skipping position due to missing keys or not being a dictionary: {pos}")


            if holdings_list:
                portfolio_holdings_df = pd.DataFrame(holdings_list)
                # Exclude zero or negative value positions as they don't contribute to weighted beta typically
                portfolio_holdings_df = portfolio_holdings_df[portfolio_holdings_df['Value'] > 0]

                if not portfolio_holdings_df.empty:
                    symbols_to_fetch = portfolio_holdings_df['Symbol'].unique().tolist()
                    market_ticker = "^GSPC" # S&P 500 as market benchmark

                    # Fetch data (e.g., for the last year)
                    end_date = date.today()
                    start_date = end_date - timedelta(days=366) # Use 366 days to ensure ~1 year of returns

                    # Add market ticker to the list for download
                    all_tickers = list(set(symbols_to_fetch + [market_ticker])) # Ensure unique tickers

                    app.logger.info(f"Fetching yfinance data for: {all_tickers}")
                    try:
                         # Download data
                         data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, timeout=30)

                         if data.empty:
                             app.logger.warning(f"yfinance download returned empty DataFrame for tickers: {all_tickers}")
                         elif 'Adj Close' not in data.columns:
                             app.logger.warning(f"'Adj Close' not found in yfinance downloaded data columns: {data.columns}")
                         else:
                             # Calculate daily returns - handle MultiIndex if multiple tickers downloaded
                             if isinstance(data.columns, pd.MultiIndex):
                                 adj_close = data['Adj Close']
                             else: # Single ticker case (unlikely here but handle)
                                 adj_close = data[['Adj Close']] # Keep as DataFrame
                                 # Rename column if it's just 'Adj Close'
                                 if len(adj_close.columns) == 1 and adj_close.columns[0] == 'Adj Close' and len(all_tickers)==1:
                                     adj_close.columns = all_tickers

                             returns = adj_close.pct_change().dropna(how='all')

                             # Check if market data exists
                             if market_ticker not in returns.columns:
                                 app.logger.warning(f"Market ticker '{market_ticker}' not found in yfinance returns columns.")
                             else:
                                 market_returns_series = returns[market_ticker].dropna()

                                 # Prepare stock returns DataFrame
                                 stock_returns_df = returns.drop(columns=[market_ticker], errors='ignore')
                                 # Filter for symbols actually in the portfolio and drop columns with all NaNs
                                 stock_returns_df = stock_returns_df[stock_returns_df.columns.intersection(symbols_to_fetch)].dropna(axis=1, how='all')

                                 if not market_returns_series.empty and not stock_returns_df.empty:
                                     app.logger.info(f"Calculating beta with {len(portfolio_holdings_df)} holdings, market data points: {len(market_returns_series)}, stock symbols in returns: {list(stock_returns_df.columns)}")
                                     # Call the calculation function
                                     portfolio_beta = calculate_portfolio_beta(
                                         portfolio_holdings=portfolio_holdings_df.copy(), # Pass a copy
                                         market_data=market_returns_series,
                                         stock_data=stock_returns_df
                                     )
                                     if portfolio_beta is not None:
                                         portfolio_beta = round(portfolio_beta, 3) # Round for display
                                         app.logger.info(f"Calculated Portfolio Beta: {portfolio_beta}")
                                     else:
                                         app.logger.warning("Portfolio beta calculation returned None.")
                                 else:
                                     app.logger.warning("Insufficient market or stock return data after processing for beta calculation.")

                    except Exception as yf_err:
                        app.logger.error(f"Error fetching or processing data from yfinance: {yf_err}", exc_info=True)
                        # Keep portfolio_beta as None
                else:
                    app.logger.info("Portfolio holdings DataFrame is empty after filtering zero/negative values.")
            else:
                 app.logger.info("No valid holdings extracted from positions data.")
        else:
             app.logger.info("No positions data received or data is not a list.")

    except requests.exceptions.RequestException as e:
         app.logger.error(f"Error fetching summary or positions for account {account_id}: {e}")
         # Don't return error page, just log it and proceed without beta
         # Summary might be empty if it failed before position fetch
         summary = summary if summary else {} # Ensure summary is a dict
    except Exception as e:
        app.logger.error(f"Unexpected error during data fetch or beta calculation: {e}", exc_info=True)
        # Log error and proceed without beta
        summary = summary if summary else {} # Ensure summary is a dict

    return render_template("dashboard.html",
                           account=account_display,
                           summary=summary,
                           portfolio_beta=portfolio_beta) # Pass beta to template


@app.route("/lookup")
def lookup():
    symbol = request.args.get('symbol', None)
    stocks = []

    if symbol is not None:
        r = requests.get(f"{BASE_API_URL}/iserver/secdef/search?symbol={symbol}&name=true", verify=False)

        response = r.json()
        stocks = response

    return render_template("lookup.html", stocks=stocks)


@app.route("/contract/<contract_id>/<period>")
def contract(contract_id, period='5d', bar='1d'):
    data = {
        "conids": [
            contract_id
        ]
    }
    
    r = requests.post(f"{BASE_API_URL}/trsrv/secdef", data=data, verify=False)
    contract = r.json()['secdef'][0]

    r = requests.get(f"{BASE_API_URL}/iserver/marketdata/history?conid={contract_id}&period={period}&bar={bar}", verify=False)
    price_history = r.json()

    return render_template("contract.html", price_history=price_history, contract=contract)


@app.route("/orders")
def orders():
    r = requests.get(f"{BASE_API_URL}/iserver/account/orders", verify=False)
    orders = r.json()["orders"]
    
    # place order code
    return render_template("orders.html", orders=orders)


@app.route("/order", methods=['POST'])
def place_order():
    print("== placing order ==")

    data = {
        "orders": [
            {
                "conid": int(request.form.get('contract_id')),
                "orderType": "LMT",
                "price": float(request.form.get('price')),
                "quantity": int(request.form.get('quantity')),
                "side": request.form.get('side'),
                "tif": "GTC"
            }
        ]
    }

    r = requests.post(f"{BASE_API_URL}/iserver/account/{ACCOUNT_ID}/orders", json=data, verify=False)

    return redirect("/orders")

@app.route("/orders/<order_id>/cancel")
def cancel_order(order_id):
    cancel_url = f"{BASE_API_URL}/iserver/account/{ACCOUNT_ID}/order/{order_id}" 
    r = requests.delete(cancel_url, verify=False)

    return r.json()


@app.route("/portfolio")
def portfolio():
    r = requests.get(f"{BASE_API_URL}/portfolio/{ACCOUNT_ID}/positions/0", verify=False)

    if r.content:
        positions = r.json()
    else:
        positions = []

    # return my positions, how much cash i have in this account
    return render_template("portfolio.html", positions=positions)

@app.route("/watchlists")
def watchlists():
    r = requests.get(f"{BASE_API_URL}/iserver/watchlists", verify=False)

    watchlist_data = r.json()["data"]
    watchlists = []
    if "user_lists" in watchlist_data:
        watchlists = watchlist_data["user_lists"]
        
    return render_template("watchlists.html", watchlists=watchlists)


@app.route("/watchlists/<int:id>")
def watchlist_detail(id):
    r = requests.get(f"{BASE_API_URL}/iserver/watchlist?id={id}", verify=False)

    watchlist = r.json()

    return render_template("watchlist.html", watchlist=watchlist)


@app.route("/watchlists/<int:id>/delete")
def watchlist_delete(id):
    r = requests.delete(f"{BASE_API_URL}/iserver/watchlist?id={id}", verify=False)

    return redirect("/watchlists")

@app.route("/watchlists/create", methods=['POST'])
def create_watchlist():
    data = request.get_json()
    name = data['name']

    rows = []
    symbols = data['symbols'].split(",")
    for symbol in symbols:
        symbol = symbol.strip()
        if symbol:
            r = requests.get(f"{BASE_API_URL}/iserver/secdef/search?symbol={symbol}&name=true&secType=STK", verify=False)
            contract_id = r.json()[0]['conid']
            rows.append({"C": contract_id})

    data = {
        "id": int(time.time()),
        "name": name,
        "rows": rows
    }

    r = requests.post(f"{BASE_API_URL}/iserver/watchlist", json=data, verify=False)
    
    return redirect("/watchlists")

@app.route("/scanner")
def scanner():
    r = requests.get(f"{BASE_API_URL}/iserver/scanner/params", verify=False)
    params = r.json()

    scanner_map = {}
    filter_map = {}

    for item in params['instrument_list']:
        scanner_map[item['type']] = {
            "display_name": item['display_name'],
            "filters": item['filters'],
            "sorts": []
        }

    for item in params['filter_list']:
        filter_map[item['group']] = {
            "display_name": item['display_name'],
            "type": item['type'],
            "code": item['code']
        }

    for item in params['scan_type_list']:
        for instrument in item['instruments']:
            scanner_map[instrument]['sorts'].append({
                "name": item['display_name'],
                "code": item['code']
            })

    for item in params['location_tree']:
        scanner_map[item['type']]['locations'] = item['locations']


    submitted = request.args.get("submitted", "")
    selected_instrument = request.args.get("instrument", "")
    location = request.args.get("location", "")
    sort = request.args.get("sort", "")
    scan_results = []
    filter_code = request.args.get("filter", "")
    filter_value = request.args.get("filter_value", "")

    if submitted:
        data = {
            "instrument": selected_instrument,
            "location": location,
            "type": sort,
            "filter": [
                {
                    "code": filter_code,
                    "value": filter_value
                }
            ]
        }
            
        r = requests.post(f"{BASE_API_URL}/iserver/scanner/run", json=data, verify=False)
        scan_results = r.json()

    return render_template("scanner.html", params=params, scanner_map=scanner_map, filter_map=filter_map, scan_results=scan_results)


@app.route("/monitor")
def monitor():
    """Render the stock price monitoring dashboard"""
    return render_template("monitor.html")

@app.route("/q_monitor")
def q_monitor():
    """Render the Q-stocks monitoring dashboard"""
    monitor = get_price_monitor()
    required_tickers = monitor.get_required_q_stocks()
    return render_template("q_monitor.html", required_tickers=required_tickers)


@app.route("/monitor/stocks")
def monitor_stocks():
    """Return all monitored stocks with their latest prices"""
    monitor = get_price_monitor()
    return jsonify({
        "success": True,
        "stocks": monitor.get_latest_prices()
    })


@app.route("/monitor/history/<conid>")
def monitor_history(conid):
    """Return price history for a specific stock"""
    import logging
    logging.info(f"Fetching history for conid: {conid}")
    
    monitor = get_price_monitor()
    data = monitor.get_stock_data(conid)
    
    if data is None:
        logging.warning(f"Stock not found: {conid}")
        return jsonify({
            "success": False,
            "message": "Stock not found"
        })
    
    # Log the amount of history data
    logging.info(f"Returning {len(data.get('history', []))} history points for {data.get('stock', {}).get('symbol')}")
    
    return jsonify({
        "success": True,
        "data": data
    })


@app.route("/monitor/add", methods=["POST"])
def monitor_add():
    """Add a stock to the monitoring list"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").strip().upper()
        
        if not symbol:
            return jsonify({
                "success": False,
                "message": "Symbol is required"
            })
            
        # Look up the symbol in IB API
        r = requests.get(f"{BASE_API_URL}/iserver/secdef/search?symbol={symbol}&name=true&secType=STK", verify=False)
        
        if r.status_code != 200:
            return jsonify({
                "success": False,
                "message": f"API error: {r.status_code}"
            })
            
        results = r.json()
        
        if not results or len(results) == 0:
            return jsonify({
                "success": False,
                "message": f"Symbol not found: {symbol}"
            })
            
        # Use the first match
        match = results[0]
        conid = str(match.get("conid"))
        name = match.get("description", "").split(" - ")[0]
        
        # Add to monitor
        monitor = get_price_monitor()
        success, message = monitor.add_stock(conid, symbol, name)
        
        return jsonify({
            "success": success,
            "message": message
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })


@app.route("/monitor/remove/<conid>", methods=["POST"])
def monitor_remove(conid):
    """Remove a stock from the monitoring list"""
    monitor = get_price_monitor()
    success, message = monitor.remove_stock(conid)
    
    return jsonify({
        "success": success,
        "message": message
    })

@app.route("/monitor/q_stocks")
def monitor_q_stocks():
    """Get Q-signal data and status"""
    monitor = get_price_monitor()
    data = monitor.get_q_signal_data()
    
    return jsonify({
        "success": True,
        "data": data
    })

@app.route("/monitor/q_signal")
def monitor_q_signal():
    """Get the current Q-signal and normalized prices"""
    monitor = get_price_monitor()
    q_data = monitor.get_q_signal_data()
    
    # Get the most recent Q-signal data point
    history = q_data.get('q_signal_history', [])
    latest = history[-1] if history else None
    
    return jsonify({
        "success": True,
        "has_all_tickers": q_data.get('has_all_tickers', False),
        "missing_tickers": q_data.get('missing_tickers', []),
        "latest": latest
    })
