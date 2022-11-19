@app.callback(
    Output('graph-content', 'children'),
    [Input('chart_selector', 'value'),Input("tabs", "active_tab"), Input('store', 'data')]
)
def render_tab_dropdown(chart, active_tab, data):
    if active_tab and data and chart is not None:
        if active_tab == "analysis":
            if chart == 'Line':
                return dcc.Graph(figure=data['line'])
            elif chart == 'Candlestick':
                return dcc.Graph(figure=data['candle'])
            elif chart == 'SMA':
                return dcc.Graph(figure=data['sma'])
            elif chart == 'EMA':
                return dcc.Graph(figure=data['ema'])
            elif chart == 'MACD':
                return dcc.Graph(figure=data['macd'])
            elif chart == 'RSI':
                return dcc.Graph(figure=data['rsi'])
            elif chart == 'OHLC':
                return dcc.Graph(figure=data['ohlc'])
            else:
                return "Hello"
        elif active_tab == "predict":
            return dcc.Graph(figure=data['predict'])
    return "Hello"


@app.callback(Output("store", "data"), [State('stock_selector', 'value')], Input("submit", "n_clicks"))
def prepare(stocks, n):
    
    df = yf.download(stocks, period='5Y')

    stock = df.copy(deep=True)
    stock = sdf.retype(stock)

    line = go.Figure(
    data=[
        go.Scatter(
            x = df.index,
            y = df['Open'],
            name = "Open"
        ),
        go.Scatter(
            x=df.index, 
            y=df['Close'], 
            name="Close"
        )
    ],
        layout={
            "height": 800,
            "title": 'Line Chart',
            "showlegend": True,
        },
    )
    
    candle = go.Figure( data = [ go.Candlestick(
        open=df['Open'],
        x=df.index,
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candlestick",
        increasing_line_color='#28C5FA',
        decreasing_line_color='#EE1755'
    )], 
        layout={
            "height": 800,
            "title": "Candlestick Chart",
            "showlegend": True,
        }
    )

    close_ma_10 = df.Close.rolling(10).mean()
    close_ma_15 = df.Close.rolling(15).mean()
    close_ma_30 = df.Close.rolling(30).mean()
    close_ma_100 = df.Close.rolling(100).mean()
    sma = go.Figure(
        data = [
            go.Scatter(
                x=list(close_ma_10.index), 
                y=list(close_ma_10), 
                name="10 Days"
            ),
            go.Scatter(
                x=list(close_ma_15.index), 
                y=list(close_ma_15), 
                name="15 Days"
            ),
            go.Scatter(
                x=list(close_ma_30.index), 
                y=list(close_ma_15), 
                name="30 Days"
            ),
            go.Scatter(
                x=list(close_ma_100.index), 
                y=list(close_ma_15), 
                name="100 Days"
            ),
        ],
        layout={
            "height": 800,
            "title": "Simple Moving Average Chart",
            "showlegend": True,
        },
    )

    ohlc = go.Figure(
    data=[
        go.Ohlc(
            x=df.index,
            open=df.Open,
            high=df.High,
            low=df.Low,
            close=df.Close,
            increasing_line_color='#28C5FA',
            decreasing_line_color='#EE1755',
            name = 'OHLC'
        )
    ],
        layout={
            "height": 800,
            "title": 'Open-High-Low-Close Chart',
            "showlegend": True,
        },
    )

    close_ema_10 = df.Close.ewm(span=10).mean()
    close_ema_15 = df.Close.ewm(span=15).mean()
    close_ema_30 = df.Close.ewm(span=30).mean()
    close_ema_100 = df.Close.ewm(span=100).mean()
    ema = go.Figure(
        data=[
            go.Scatter(
                x=list(close_ema_10.index), 
                y=list(close_ema_10), 
                name="10 Days"
            ),
            go.Scatter(
                x=list(close_ema_15.index), 
                y=list(close_ema_15), 
                name="15 Days"
            ),
            go.Scatter(
                x=list(close_ema_30.index), 
                y=list(close_ema_30), 
                name="30 Days"
            ),
            go.Scatter(
                x=list(close_ema_100.index),
                y=list(close_ema_100),
                name="100 Days",
            ),
        ],
        layout={
            "height": 800,
            "title": "Exponential Moving Average Chart",
            "showlegend": True,
        },
    )

    macd, macds, macdh = stock["macd"], stock["macds"], stock["macdh"]
    macd = go.Figure( data=[
            go.Scatter(
                x=list(stock.index), 
                y=list(macd), 
                name="MACD"),
            go.Scatter(
                x=list(stock.index), 
                y=list(macds), 
                name="Signal"
            ),
            go.Scatter(
                x=list(stock.index),
                y=list(macdh),
                line=dict(color= "rgba(255,144,162,0.23)", width=4, dash="dot"),
                name="Histogram",
            ),
        ],
        layout={
            "height": 800,
            "title": "Moving Average Convergence Divergence Chart",
            "showlegend": True,
        },
    )

    rsi_6 = stock["rsi_6"]
    rsi_12 = stock["rsi_12"]
    rsi = go.Figure(
        data=[
            go.Scatter(
                x=list(stock.index), 
                y=list(rsi_6), 
                name="RSI 6 Day"
            ),
            go.Scatter(
                x=list(df.index), 
                y=list(rsi_12), 
                name="RSI 12 Day"
            ),
        ],
        layout={
            "height": 800,
            "title": 'Relative Strength Index Chart',
            "showlegend": True,
        },
    )


    start = datetime(2018, 1, 1)
    end = datetime.now() - timedelta(int(datetime.now().timetuple().tm_yday))

    dataset_train = yf.download(stocks, start=start, end=end)
    dataset_train.drop(columns=['Adj Close', 'Volume', "High", "Low"], axis=1, inplace=True)

    training_set = dataset_train.iloc[:, 1:2].values

    scale = MinMaxScaler(feature_range=(0,1))

    training_set_scaled = scale.fit_transform(training_set)

    X_train = []
    y_train = []

    for i in range(60, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-60: i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, newshape = (X_train.shape[0], X_train.shape[1], 1))

    regressor = Sequential()

    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(rate = 0.2))

    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(rate = 0.2))
    regressor.add(LSTM(units = 50, return_sequences = False))
    regressor.add(Dropout(rate = 0.2))
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    regressor.fit(x = X_train, y = y_train, batch_size = 32, epochs = 50)

    start_2 = datetime.now() - timedelta(313)
    end2 = datetime.now()

    dataset_test = yf.download(stocks, start = start_2, end = end2)
    dataset_test.drop(columns = ['Adj Close', 'Volume', "High", "Low"], axis = 1, inplace = True)

    dataset_total = pd.concat((dataset_train['Open'],dataset_test['Open']), axis = 0)

    inputs = dataset_total[len(dataset_total)-len(dataset_test)- 60: ].values
    inputs = inputs.reshape(-1, 1)
    inputs = scale.transform(inputs)
    X_test = []

    for i in range(60, len(inputs)): 
        X_test.append(inputs[i-60: i, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, newshape = (X_test.shape[0],  X_test.shape[1], 1))
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = scale.inverse_transform(predicted_stock_price)

    test_df_price_predict = pd.DataFrame(predicted_stock_price, columns = ['Open'])
    test_df_price_predict.index = dataset_test.index


    trace_1 = go.Scatter(
        x = dataset_train.index,
        y = dataset_train['Open'],
        name = 'Trained Price'
    )
    
    trace_2 = go.Scatter(
        x = dataset_test.index,
        y = dataset_test['Open'],
        name = "Test Price"
    )

    predict = go.Figure(
        data=[trace_1, trace_2]
    )

    return {"line": line, "candle": candle, "sma": sma, "ohlc": ohlc, "ema": ema, "macd": macd, "rsi": rsi, "predict": predict}

if __name__ == "__main__":
    app.run_server(debug=True)  