from dash import Dash, html, dcc, Input, Output, State, callback_context
from datetime import date, datetime
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Initialize Dash app with external CSS
app = Dash(__name__, external_stylesheets=['/assets/styles.css'])
app.title = "StockLens"

# Copyright and disclaimer
COPYRIGHT_INFO = """
© 2025 StockLens. All rights reserved.
Data provided by Yahoo Finance. For educational purposes only.
Not financial advice. Please consult a financial advisor before making investment decisions.
"""

# Color scheme for charts
COLORS = {
    "background": "#ffffff",
    "grid": "#e5e5e5", 
    "text": "#333333",
    "up": "#2ecc71",
    "down": "#e74c3c",
    "sma20": "#3498db",
    "sma50": "#f39c12", 
    "sma100": "#9b59b6",
    "sma200": "#34495e",
    "ema200": "#d35400",
    "stoch_k": "#e74c3c",
    "stoch_d": "#3498db",
    "overbought": "#ff6b6b",
    "oversold": "#4ecdc4"
}

def create_empty_figure(title=""):
    """Create an empty figure with consistent styling"""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(color=COLORS['text']),
        yaxis=dict(title='Price', gridcolor=COLORS['grid']),
        xaxis=dict(gridcolor=COLORS['grid'])
    )
    return fig

def calculate_rsi(close_prices, period=14):
    """Calculate RSI (Relative Strength Index)"""
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stochastic_rsi(close_prices, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    """Calculate Stochastic RSI"""
    # Calculate RSI
    rsi = calculate_rsi(close_prices, rsi_period)
    
    # Calculate Stochastic of RSI
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    
    # Avoid division by zero
    rsi_range = rsi_max - rsi_min
    stoch_rsi = np.where(rsi_range != 0, (rsi - rsi_min) / rsi_range * 100, 50)
    stoch_rsi = pd.Series(stoch_rsi, index=close_prices.index)
    
    # Smooth %K and calculate %D
    stoch_k = stoch_rsi.rolling(window=k_smooth).mean()
    stoch_d = stoch_k.rolling(window=d_smooth).mean()
    
    return stoch_k, stoch_d, rsi

def calculate_daily_changes_and_streaks(df):
    """Calculate daily changes, consecutive streaks, and movement statistics"""
    df = df.copy()
    
    # Calculate daily changes (absolute and percentage)
    df['Daily_Change'] = df['Close'] - df['Open']
    df['Daily_Change_Pct'] = (df['Daily_Change'] / df['Open']) * 100
    
    # Determine if candle is green (bullish) or red (bearish)
    df['Is_Green'] = df['Close'] > df['Open']
    df['Is_Red'] = df['Close'] < df['Open']
    
    # Calculate consecutive green and red candles
    df['Green_Streak'] = 0
    df['Red_Streak'] = 0
    
    green_count = 0
    red_count = 0
    
    for i in range(len(df)):
        if df['Is_Green'].iloc[i]:
            green_count += 1
            red_count = 0
        elif df['Is_Red'].iloc[i]:
            red_count += 1
            green_count = 0
        else:  # Doji (open == close)
            green_count = 0
            red_count = 0
        
        df.iloc[i, df.columns.get_loc('Green_Streak')] = green_count
        df.iloc[i, df.columns.get_loc('Red_Streak')] = red_count
    
    return df

def calculate_movement_statistics(df):
    """Calculate movement statistics for positive and negative moves"""
    # Separate positive and negative movements
    positive_moves = df[df['Daily_Change'] > 0]['Daily_Change']
    negative_moves = df[df['Daily_Change'] < 0]['Daily_Change']
    positive_moves_pct = df[df['Daily_Change_Pct'] > 0]['Daily_Change_Pct']
    negative_moves_pct = df[df['Daily_Change_Pct'] < 0]['Daily_Change_Pct']
    
    # Calculate statistics
    stats = {
        'positive_count': len(positive_moves),
        'negative_count': len(negative_moves),
        'positive_max': positive_moves.max() if len(positive_moves) > 0 else 0,
        'positive_min': positive_moves.min() if len(positive_moves) > 0 else 0,
        'positive_avg': positive_moves.mean() if len(positive_moves) > 0 else 0,
        'negative_max': abs(negative_moves.min()) if len(negative_moves) > 0 else 0,  # Most negative (largest drop)
        'negative_min': abs(negative_moves.max()) if len(negative_moves) > 0 else 0,  # Least negative (smallest drop)
        'negative_avg': abs(negative_moves.mean()) if len(negative_moves) > 0 else 0,
        'positive_max_pct': positive_moves_pct.max() if len(positive_moves_pct) > 0 else 0,
        'positive_min_pct': positive_moves_pct.min() if len(positive_moves_pct) > 0 else 0,
        'positive_avg_pct': positive_moves_pct.mean() if len(positive_moves_pct) > 0 else 0,
        'negative_max_pct': abs(negative_moves_pct.min()) if len(negative_moves_pct) > 0 else 0,
        'negative_min_pct': abs(negative_moves_pct.max()) if len(negative_moves_pct) > 0 else 0,
        'negative_avg_pct': abs(negative_moves_pct.mean()) if len(negative_moves_pct) > 0 else 0,
        'max_green_streak': df['Green_Streak'].max(),
        'max_red_streak': df['Red_Streak'].max(),
        'total_days': len(df),
        'green_days': len(positive_moves),
        'red_days': len(negative_moves),
        'doji_days': len(df) - len(positive_moves) - len(negative_moves)
    }
    
    return stats

def calculate_moving_averages(df):
    """Calculate moving averages for the dataframe"""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean() 
    df['SMA_100'] = df['Close'].rolling(window=100, min_periods=1).mean()
    df['SMA_200'] = df['Close'].rolling(window=200, min_periods=1).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Add Stochastic RSI
    stoch_k, stoch_d, rsi = calculate_stochastic_rsi(df['Close'])
    df['Stoch_RSI_K'] = stoch_k
    df['Stoch_RSI_D'] = stoch_d
    df['RSI'] = rsi
    
    # Add daily changes and streaks
    df = calculate_daily_changes_and_streaks(df)
    
    return df

def create_candlestick_chart(df, title, include_sma=True, include_ema=False):
    """Create candlestick chart with optional moving averages"""
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'], 
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color=COLORS['up'],
        decreasing_line_color=COLORS['down']
    ))
    
    if include_sma:
        # Add SMA traces
        sma_configs = [
            (df['SMA_20'], COLORS['sma20'], 'SMA 20'),
            (df['SMA_50'], COLORS['sma50'], 'SMA 50'),
            (df['SMA_100'], COLORS['sma100'], 'SMA 100'),
            (df['SMA_200'], COLORS['sma200'], 'SMA 200')
        ]
        
        for sma_data, color, name in sma_configs:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=sma_data,
                mode='lines',
                name=name,
                line=dict(width=2, color=color),
                opacity=0.8
            ))
    
    if include_ema:
        # Add EMA trace
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['EMA_200'],
            mode='lines',
            name='EMA 200',
            line=dict(width=2, dash='dash', color=COLORS['ema200']),
            opacity=0.8
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis_rangeslider_visible=(title.find('EMA') != -1),
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(color=COLORS['text']),
        yaxis=dict(title='Price', gridcolor=COLORS['grid']),
        xaxis=dict(title='Date', gridcolor=COLORS['grid']),
        hovermode='x unified'
    )
    
    return fig

def create_stoch_rsi_chart(df, title):
    """Create Stochastic RSI chart"""
    fig = go.Figure()
    
    # Add Stochastic RSI %K line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Stoch_RSI_K'],
        mode='lines',
        name='Stoch RSI %K',
        line=dict(width=2, color=COLORS['stoch_k']),
        opacity=0.8
    ))
    
    # Add Stochastic RSI %D line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Stoch_RSI_D'],
        mode='lines',
        name='Stoch RSI %D',
        line=dict(width=2, color=COLORS['stoch_d']),
        opacity=0.8
    ))
    
    # Add overbought line (80)
    fig.add_hline(
        y=80, 
        line_dash="dash", 
        line_color=COLORS['overbought'],
        annotation_text="Overbought (80)",
        annotation_position="bottom right"
    )
    
    # Add oversold line (20)
    fig.add_hline(
        y=20, 
        line_dash="dash", 
        line_color=COLORS['oversold'],
        annotation_text="Oversold (20)",
        annotation_position="top right"
    )
    
    # Add middle line (50)
    fig.add_hline(
        y=50, 
        line_dash="dot", 
        line_color=COLORS['grid'],
        opacity=0.5
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        template='plotly_white',
        xaxis_rangeslider_visible=True,
        height=350,
        margin=dict(t=50, b=50, l=50, r=50),
        font=dict(color=COLORS['text']),
        yaxis=dict(
            title='Stochastic RSI (%)', 
            gridcolor=COLORS['grid'],
            range=[0, 100]
        ),
        xaxis=dict(title='Date', gridcolor=COLORS['grid']),
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98)
    )
    
    return fig

def validate_inputs(ticker, start_date, end_date):
    """Validate user inputs and return error message if any"""
    if not ticker or ticker.strip() == "":
        return "⚠️ Please enter a ticker symbol."
    
    if not start_date or not end_date:
        return "⚠️ Please select both start and end dates."
    
    try:
        start_dt = datetime.fromisoformat(start_date).date()
        end_dt = datetime.fromisoformat(end_date).date()
        
        if start_dt > end_dt:
            return "⚠️ Start date cannot be after end date."
        
        if end_dt > date.today():
            return "⚠️ End date cannot be in the future."
            
        if start_dt < date(1980, 1, 1):
            return "⚠️ Start date too early. Please select a date after 1980."
            
    except (ValueError, TypeError):
        return "⚠️ Invalid date format."
    
    return None

def create_stock_info_display(ticker, company_name, sector, exchange, start_date, end_date, df, stats):
    """Create the stock information display component with enhanced metrics"""
    first_price = df['Close'].iloc[0]
    last_price = df['Close'].iloc[-1]
    price_change = last_price - first_price
    price_change_pct = (price_change / first_price) * 100
    
    change_color = COLORS['up'] if price_change >= 0 else COLORS['down']
    change_symbol = "📈" if price_change >= 0 else "📉"
    
    # Get latest Stochastic RSI values
    latest_stoch_k = df['Stoch_RSI_K'].iloc[-1] if not pd.isna(df['Stoch_RSI_K'].iloc[-1]) else 0
    latest_stoch_d = df['Stoch_RSI_D'].iloc[-1] if not pd.isna(df['Stoch_RSI_D'].iloc[-1]) else 0
    latest_rsi = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 0
    
    # Determine signal
    if latest_stoch_k > 80:
        signal = "🔴 Overbought"
        signal_color = COLORS['overbought']
    elif latest_stoch_k < 20:
        signal = "🟢 Oversold"
        signal_color = COLORS['oversold']
    else:
        signal = "🟡 Neutral"
        signal_color = COLORS['text']
    
    # Calculate win rate
    win_rate = (stats['positive_count'] / stats['total_days']) * 100 if stats['total_days'] > 0 else 0
    
    return html.Div(
        className="stock-info",
        children=[
            html.H4(f"{ticker}"),
            html.P(f"🏢 Company: {company_name}"),
            html.P(f"🏦 Exchange: {exchange}"),
            html.P(f"🏭 Sector: {sector}"),
            html.Hr(),
            html.P(f"📅 Period: {start_date} to {end_date}"),
            html.P(f"📊 Data Points: {len(df)}"),
            html.P(f"💰 Start Price: {first_price:.2f}"),
            html.P(f"💰 End Price: {last_price:.2f}"),
            html.P([
                f"{change_symbol} Change: ",
                html.Span(
                    f"{price_change:+.2f} ({price_change_pct:+.1f}%)", 
                    style={'color': change_color, 'fontWeight': 'bold'}
                )
            ]),
            html.Hr(),
            html.H5("📈 Technical Indicators"),
            html.P(f"RSI: {latest_rsi:.1f}"),
            html.P(f"Stoch RSI %K: {latest_stoch_k:.1f}"),
            html.P(f"Stoch RSI %D: {latest_stoch_d:.1f}"),
            html.P([
                "Signal: ",
                html.Span(
                    signal,
                    style={'color': signal_color, 'fontWeight': 'bold'}
                )
            ]),
            html.Hr(),
            html.H5("📊 Daily Movement Stats"),
            html.P(f"🟢 Green Days: {stats['green_days']} ({win_rate:.1f}%)"),
            html.P(f"🔴 Red Days: {stats['red_days']} ({(100-win_rate):.1f}%)"),
            html.P(f"⚪ Doji Days: {stats['doji_days']}"),
            html.P(f"🔥 Max Green Streak: {stats['max_green_streak']} days"),
            html.P(f"❄️ Max Red Streak: {stats['max_red_streak']} days"),
            html.Hr(),
            html.H5("💹 Positive Moves"),
            html.P(f"🎯 Max Gain: ₹{stats['positive_max']:.2f} ({stats['positive_max_pct']:.2f}%)"),
            html.P(f"📈 Min Gain: ₹{stats['positive_min']:.2f} ({stats['positive_min_pct']:.2f}%)"),
            html.P(f"📊 Avg Gain: ₹{stats['positive_avg']:.2f} ({stats['positive_avg_pct']:.2f}%)"),
            html.Hr(),
            html.H5("📉 Negative Moves"),
            html.P(f"💥 Max Loss: ₹{stats['negative_max']:.2f} ({stats['negative_max_pct']:.2f}%)"),
            html.P(f"📉 Min Loss: ₹{stats['negative_min']:.2f} ({stats['negative_min_pct']:.2f}%)"),
            html.P(f"📊 Avg Loss: ₹{stats['negative_avg']:.2f} ({stats['negative_avg_pct']:.2f}%)"),
        ]
    )

# App Layout
app.layout = html.Div(
    className="dash-container",
    children=[
        # Left Panel - Controls
        html.Div(
            className="left-panel",
            children=[
                html.H2("StockLens", className="dashboard-title"),
                
                # Ticker Input
                html.Div(
                    className="input-group",
                    children=[
                        html.Label("Ticker Symbol", className="input-label"),
                        dcc.Input(
                            id='ticker-input',
                            type='text',
                            placeholder='e.g., AAPL, GOOGL, MSFT',
                            value='AAPL',
                            className="ticker-input",
                            debounce=True
                        )
                    ]
                ),
                
                # Date Range Picker
                html.Div(
                    className="input-group",
                    children=[
                        html.Label("Date Range", className="input-label"),
                        html.Div(
                            className="date-picker-container",
                            children=[
                                dcc.DatePickerRange(
                                    id='date-picker-range',
                                    min_date_allowed=date(1980, 1, 1),
                                    max_date_allowed=date.today(),
                                    initial_visible_month=date.today(),
                                    start_date=date(2023, 1, 1),
                                    end_date=date.today(),
                                    display_format='MMM D, YYYY'
                                )
                            ]
                        )
                    ]
                ),
                
                # Submit Button
                html.Button(
                    '🔍 Analyze Stock',
                    id='submit-button',
                    n_clicks=0,
                    className="submit-button"
                ),
                
                # Info Display
                html.Div(id='stock-info'),
                
                # Copyright Section
                html.Hr(style={'marginTop': '30px'}),
                html.Div(
                    className="copyright-section",
                    children=[
                        html.P(
                            "© 2025 StockLens",
                            style={
                                'fontSize': '12px',
                                'color': '#666',
                                'textAlign': 'center',
                                'marginBottom': '5px'
                            }
                        ),
                        html.P(
                            "Data: Educational Use Only",
                            style={
                                'fontSize': '11px',
                                'color': '#888',
                                'textAlign': 'center',
                                'marginBottom': '5px'
                            }
                        ),
                        html.P(
                            "⚠️ Not Financial Advice",
                            style={
                                'fontSize': '11px',
                                'color': '#e74c3c',
                                'textAlign': 'center',
                                'fontWeight': 'bold',
                                'marginBottom': '0px'
                            }
                        )
                    ]
                )
            ]
        ),
        
        # Right Panel - Charts
        html.Div(
            className="right-panel",
            children=[
                html.Div(
                    className="charts-container",
                    children=[
                        html.Div(
                            className="chart-container",
                            children=[
                                dcc.Graph(
                                    id='sma-chart',
                                    figure=create_empty_figure("Select a stock to view SMA analysis")
                                )
                            ]
                        ),
                        html.Div(
                            className="chart-container",
                            children=[
                                dcc.Graph(
                                    id='ema-chart', 
                                    figure=create_empty_figure("Select a stock to view EMA analysis")
                                )
                            ]
                        ),
                        html.Div(
                            className="chart-container",
                            children=[
                                dcc.Graph(
                                    id='stoch-rsi-chart',
                                    figure=create_empty_figure("Select a stock to view Stochastic RSI")
                                )
                            ]
                        )
                    ]
                )
            ]
        )
    ]
)

# Main callback for updating charts and info
@app.callback(
    [Output('stock-info', 'children'),
     Output('sma-chart', 'figure'),
     Output('ema-chart', 'figure'),
     Output('stoch-rsi-chart', 'figure')],
    [Input('submit-button', 'n_clicks')],
    [State('ticker-input', 'value'),
     State('date-picker-range', 'start_date'),
     State('date-picker-range', 'end_date')]
)
def update_dashboard(n_clicks, ticker, start_date, end_date):
    """Main callback to update the dashboard"""
    
    # Return empty state on initial load
    if n_clicks == 0:
        return ("", 
                create_empty_figure("Click 'Analyze Stock' to get started"), 
                create_empty_figure("Click 'Analyze Stock' to get started"),
                create_empty_figure("Click 'Analyze Stock' to get started"))
    
    # Validate inputs
    error_msg = validate_inputs(ticker, start_date, end_date)
    if error_msg:
        return (html.Div(error_msg, className="error-message"), 
                create_empty_figure("Error"), 
                create_empty_figure("Error"),
                create_empty_figure("Error"))
    
    try:
        # Clean ticker symbol
        ticker = ticker.strip().upper()
        
        # Fetch stock data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            error_msg = f"❌ No data found for {ticker} in the selected date range."
            return (html.Div(error_msg, className="error-message"), 
                    create_empty_figure("No Data"), 
                    create_empty_figure("No Data"),
                    create_empty_figure("No Data"))
        
        # Get stock info
        try:
            info = stock.info
            company_name = info.get('longName', info.get('shortName', 'N/A'))
            sector = info.get('sector', 'N/A')
            exchange = info.get('exchange', 'N/A')
        except:
            company_name = 'N/A'
            sector = 'N/A' 
            exchange = 'N/A'
        
        # Calculate moving averages and technical indicators
        df = calculate_moving_averages(df)
        
        # Calculate movement statistics
        stats = calculate_movement_statistics(df)
        
        # Create stock info display with enhanced metrics
        stock_info = create_stock_info_display(
            ticker, company_name, sector, exchange, start_date, end_date, df, stats
        )
        
        # Create charts
        sma_fig = create_candlestick_chart(df, f"{ticker} - Candlestick with SMAs", include_sma=True)
        sma_fig.update_layout(xaxis_rangeslider_visible=True)
        ema_fig = create_candlestick_chart(df, f"{ticker} - Candlestick with EMA 200", include_sma=False, include_ema=True)
        stoch_rsi_fig = create_stoch_rsi_chart(df, f"{ticker} - Stochastic RSI")
        
        return stock_info, sma_fig, ema_fig, stoch_rsi_fig
        
    except Exception as e:
        error_msg = f"❌ Error fetching data: {str(e)}"
        return (html.Div(error_msg, className="error-message"), 
                create_empty_figure("Error"), 
                create_empty_figure("Error"),
                create_empty_figure("Error"))

# Callback for synchronizing zoom between charts
@app.callback(
    [Output('sma-chart', 'figure', allow_duplicate=True),
     Output('ema-chart', 'figure', allow_duplicate=True),
     Output('stoch-rsi-chart', 'figure', allow_duplicate=True)],
    [Input('sma-chart', 'relayoutData'),
     Input('ema-chart', 'relayoutData'),
     Input('stoch-rsi-chart', 'relayoutData')],
    [State('sma-chart', 'figure'),
     State('ema-chart', 'figure'),
     State('stoch-rsi-chart', 'figure')],
    prevent_initial_call=True
)
def sync_chart_zoom(sma_relayout, ema_relayout, stoch_relayout, sma_fig, ema_fig, stoch_fig):
    """Synchronize zoom levels between the three charts"""
    
    ctx = callback_context
    if not ctx.triggered:
        return sma_fig, ema_fig, stoch_fig
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    # Determine which chart triggered the callback
    if triggered_id == 'sma-chart':
        relayout_data = sma_relayout
    elif triggered_id == 'ema-chart':
        relayout_data = ema_relayout
    else:
        relayout_data = stoch_relayout
    
    if not relayout_data:
        return sma_fig, ema_fig, stoch_fig
    
    # Copy figures to avoid mutation
    sma_fig_copy = sma_fig.copy() if sma_fig else create_empty_figure()
    ema_fig_copy = ema_fig.copy() if ema_fig else create_empty_figure()
    stoch_fig_copy = stoch_fig.copy() if stoch_fig else create_empty_figure()
    
    # Handle zoom synchronization
    if 'xaxis.range[0]' in relayout_data and 'xaxis.range[1]' in relayout_data:
        x_range = [relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
        sma_fig_copy['layout']['xaxis']['range'] = x_range
        ema_fig_copy['layout']['xaxis']['range'] = x_range
        stoch_fig_copy['layout']['xaxis']['range'] = x_range
    
    # Handle auto-range reset
    elif 'xaxis.autorange' in relayout_data and relayout_data['xaxis.autorange']:
        for fig_copy in [sma_fig_copy, ema_fig_copy, stoch_fig_copy]:
            if 'range' in fig_copy['layout']['xaxis']:
                del fig_copy['layout']['xaxis']['range']
    
    return sma_fig_copy, ema_fig_copy, stoch_fig_copy

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8050)