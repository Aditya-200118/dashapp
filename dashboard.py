path = os.getcwd() + "\\tickers.pickle"
with open(path, 'rb') as f:
    ticker_list = pickle.load(f)

app = Dash(__name__)

button = dbc.Button(
    'Plot',
    id='submit',
    className = 'd-grid gap-2 col-6 mx-auto mt-3 mb-3',
    n_clicks=1,
)


dropdown = dcc.Dropdown(
    id="stock_selector",
    className="mb-3 mt-4",
    options=[
        {
            "label": str(ticker_list[i]),
            "value": str(ticker_list[i]),
        }
        for i in range(len(ticker_list))
    ],
    searchable=True,
    multi=False,
    value=['TSLA'],
    placeholder="Enter Stock Name",
)

indicators = dcc.Dropdown(
    id="chart_selector",
    className='mt-3',
    options=[
        {"label": "Line", "value": "Line"},
        {"label": "Candlestick", "value": "Candlestick"},
        {"label": "Simple moving average", "value": "SMA"},
        {"label": "Exponential moving average", "value": "EMA", },
        {"label": "MACD", "value": "MACD"},
        {"label": "RSI", "value": "RSI"},
        {"label": "OHLC", "value": "OHLC"},
    ],
    value="Line",
)

app.layout = dbc.Container(
    id='container',
    children=[
        dcc.Store(id="store"),
        dbc.Row(
            children = [
            dropdown,
            indicators,
            button,
            ]
        ),
        dbc.Row(
            dbc.Col(
                sm=12,
                md=12,
                lg=12,
                width=6,
                children=dbc.Tabs(
                    [
                        dbc.Tab(label="Analysis", tab_id="analysis",
                                tab_class_name="flex-grow-1 text-center fw-bold", active_label_style={"color": "#101010"}),
                        dbc.Tab(label="Predict", tab_id="predict",
                                tab_class_name="flex-grow-1 text-center fw-bold", active_label_style={"color": "#101010"}),
                    ],
                    id="tabs",
                    active_tab="analysis",
                ),
                style={"border": "1px solid #dddddd",
                       "border-radius": "3px", "padding": "10px 20px"}
            ),
        ),
        html.Div(id="graph-content", className="p-4"),
    ], 
    fluid=True,
    style={
        "background-color": "#1A1B1E"
    }
)
