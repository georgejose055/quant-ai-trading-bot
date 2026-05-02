import warnings
warnings.filterwarnings("ignore")

import os
import dash
from dash import dcc, html, Input, Output, State, dash_table, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
from groq import Groq
import traceback

from model import fetch_data, engineer_features, train_model, get_signals

# ─── Groq Client ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
groq_client  = Groq(api_key=GROQ_API_KEY)

# ─── Dash App ─────────────────────────────────────────────────────────────────
app    = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server   # Required for gunicorn / Render deployment
app.title = "Quant-AI Trading Bot"

# ─── Layout ───────────────────────────────────────────────────────────────────
app.layout = dbc.Container([

    dbc.Row([
        dbc.Col(html.H2("📈 Quant-AI Trading Bot", className="text-center my-3",
            style={
                "background": "linear-gradient(135deg, #7c3aed, #3b82f6)",
                "WebkitBackgroundClip": "text",
                "WebkitTextFillColor": "transparent",
                "backgroundClip": "text",
                "fontWeight": "700",
                "letterSpacing": "1px",
                "fontSize": "1.8rem"
            }))
    ]),

    dbc.Row([
        dbc.Col(html.Div(id="error-alert", style={"display": "none"}), width=12)
    ]),

    dbc.Tabs([

        # ════════════════════════════════════════════════════════════════
        # TAB 1 — TRADING DASHBOARD
        # ════════════════════════════════════════════════════════════════
        dbc.Tab(label="📊 Trading Dashboard", tab_id="tab-dashboard", children=[

            dbc.Row([
                dbc.Col([
                    dbc.Label("🔍 Search Any Stock (name or ticker)", style={
                        "color": "#a78bfa", "fontSize": "0.78rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px"
                    }),
                    dcc.Input(
                        id="search-input", type="text",
                        placeholder="e.g. Apple, NVDA, Reliance, Zomato...",
                        debounce=True,
                        style={
                            "width": "100%", "borderRadius": "8px",
                            "padding": "8px 12px", "fontSize": "0.9rem",
                            "backgroundColor": "#1e1e2e", "color": "white",
                            "border": "1px solid #3d3d5c"
                        }
                    ),
                    html.Div(id="search-results-dropdown", style={"marginTop": "4px"})
                ], xs=12, sm=12, md=5, lg=5),

                dbc.Col([
                    dbc.Label("Selected Ticker", style={
                        "color": "#a78bfa", "fontSize": "0.78rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px"
                    }),
                    html.Div(id="selected-ticker-display", children="NVDA",
                        style={"color": "#facc15", "fontWeight": "700",
                               "fontSize": "1.1rem", "padding": "8px 0"}),
                    dcc.Store(id="selected-ticker-store", data="NVDA"),
                ], xs=12, sm=6, md=3, lg=3),

                dbc.Col([
                    dbc.Label("Select Period", style={
                        "color": "#a78bfa", "fontSize": "0.78rem",
                        "textTransform": "uppercase", "letterSpacing": "0.5px"
                    }),
                    dcc.Dropdown(
                        id="period-dropdown",
                        options=[
                            {"label": "3 Months", "value": "3mo"},
                            {"label": "6 Months", "value": "6mo"},
                            {"label": "1 Year",   "value": "1y"},
                            {"label": "2 Years",  "value": "2y"},
                        ],
                        value="1y", clearable=False,
                        style={"color": "#000", "borderRadius": "8px", "fontSize": "0.9rem"}
                    )
                ], xs=12, sm=6, md=2, lg=2),

                dbc.Col([
                    dbc.Label("‎", style={"display": "block"}),
                    dbc.Button("Run Model 🚀", id="run-btn", className="w-100",
                        style={
                            "background": "linear-gradient(135deg, #7c3aed, #3b82f6)",
                            "border": "none", "borderRadius": "8px",
                            "fontWeight": "600", "fontSize": "0.95rem",
                            "padding": "10px",
                            "boxShadow": "0 4px 15px rgba(124,58,237,0.4)"
                        })
                ], xs=12, sm=12, md=2, lg=2),
            ], className="mb-4 g-3 mt-3"),

            dbc.Row(id="kpi-cards", className="mb-4 g-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="price-chart",
                    config={"displayModeBar": True, "responsive": True},
                    style={"height": "500px"}), width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="rsi-chart",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "280px"}), width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="macd-chart",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "280px"}), width=12)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col(dcc.Graph(id="feature-importance-chart",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "400px"}), xs=12, sm=12, md=6, lg=6),
                dbc.Col(dcc.Graph(id="signal-dist-chart",
                    config={"displayModeBar": False, "responsive": True},
                    style={"height": "400px"}), xs=12, sm=12, md=6, lg=6),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        dbc.Col(html.H5("📋 Latest 10 Trading Signals", style={
                            "color": "#a78bfa", "fontWeight": "600", "marginBottom": "12px"
                        })),
                        dbc.Col(dbc.Button("⬇️ Download CSV", id="download-btn", size="sm",
                            style={"background": "#1e1e2e", "border": "1px solid #7c3aed",
                                   "color": "#a78bfa", "borderRadius": "8px"}),
                            width="auto"),
                    ], align="center"),
                    dcc.Download(id="download-csv"),
                    dash_table.DataTable(
                        id="signals-table",
                        style_table={"overflowX": "auto", "borderRadius": "12px",
                                     "overflow": "hidden", "width": "100%"},
                        style_cell={"backgroundColor": "#1e1e2e", "color": "white",
                                    "textAlign": "center", "padding": "10px 16px",
                                    "border": "none", "fontSize": "0.88rem",
                                    "maxWidth": "160px", "overflow": "hidden",
                                    "textOverflow": "ellipsis"},
                        style_header={"backgroundColor": "#7c3aed", "fontWeight": "bold",
                                      "color": "white", "textTransform": "uppercase",
                                      "fontSize": "0.75rem", "letterSpacing": "0.5px",
                                      "border": "none", "padding": "12px 16px"},
                        style_data_conditional=[
                            {"if": {"filter_query": '{Signal} = "BUY 🟢"'},
                             "color": "#4ade80", "fontWeight": "600"},
                            {"if": {"filter_query": '{Signal} = "SELL 🔴"'},
                             "color": "#f87171", "fontWeight": "600"},
                            {"if": {"row_index": "odd"}, "backgroundColor": "#16162a"}
                        ],
                    )
                ], width=12)
            ], className="mt-2 mb-5"),

        ]),

        # ════════════════════════════════════════════════════════════════
        # TAB 2 — AI CHATBOT
        # ════════════════════════════════════════════════════════════════
        dbc.Tab(label="🤖 AI Investment Advisor", tab_id="tab-chatbot", children=[

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("💬 AI Investment Chatbot", style={
                            "color": "#a78bfa", "fontWeight": "600", "marginBottom": "4px"
                        }),
                        html.P(
                            "Ask anything — I scan 200+ stocks (large, mid, small cap) "
                            "across NSE & US and detect ₹ vs $ automatically.",
                            style={"color": "#64748b", "fontSize": "0.82rem", "marginBottom": "16px"}
                        ),

                        html.Div(id="chat-history", style={
                            "backgroundColor": "#13131f",
                            "borderRadius": "12px",
                            "padding": "16px",
                            "minHeight": "420px",
                            "maxHeight": "420px",
                            "overflowY": "auto",
                            "marginBottom": "16px",
                            "border": "1px solid #2d2d44",
                            "display": "flex",
                            "flexDirection": "column",
                            "gap": "12px"
                        }),

                        dcc.Store(id="chat-store", data=[]),

                        dbc.Row([
                            dbc.Col(
                                dcc.Input(
                                    id="chat-input", type="text",
                                    placeholder="Type your question and press Enter...",
                                    debounce=False, n_submit=0,
                                    style={
                                        "width": "100%", "borderRadius": "8px",
                                        "padding": "10px 14px", "fontSize": "0.92rem",
                                        "backgroundColor": "#1e1e2e", "color": "white",
                                        "border": "1px solid #3d3d5c"
                                    }
                                ), width=10
                            ),
                            dbc.Col(
                                dbc.Spinner(
                                    dbc.Button("Send 🚀", id="chat-send-btn", className="w-100",
                                        style={
                                            "background": "linear-gradient(135deg, #7c3aed, #3b82f6)",
                                            "border": "none", "borderRadius": "8px",
                                            "fontWeight": "600", "fontSize": "0.9rem",
                                            "padding": "10px"
                                        }),
                                    color="light", size="sm"
                                ), width=2
                            ),
                        ], className="g-2"),

                        html.Div([
                            html.P("💡 Quick Prompts:", style={
                                "color": "#64748b", "fontSize": "0.78rem",
                                "marginTop": "12px", "marginBottom": "6px"
                            }),
                            dbc.Row([
                                dbc.Col(dbc.Button("I have ₹3000, make 10x in 1 year?",
                                    id="quick-1", size="sm", outline=True,
                                    style={"fontSize": "0.75rem", "borderRadius": "20px",
                                           "border": "1px solid #3d3d5c", "color": "#94a3b8"}
                                ), width="auto"),
                                dbc.Col(dbc.Button("Best Indian stocks under ₹500?",
                                    id="quick-2", size="sm", outline=True,
                                    style={"fontSize": "0.75rem", "borderRadius": "20px",
                                           "border": "1px solid #3d3d5c", "color": "#94a3b8"}
                                ), width="auto"),
                                dbc.Col(dbc.Button("Best US small cap stocks right now?",
                                    id="quick-3", size="sm", outline=True,
                                    style={"fontSize": "0.75rem", "borderRadius": "20px",
                                           "border": "1px solid #3d3d5c", "color": "#94a3b8"}
                                ), width="auto"),
                            ], className="g-2"),
                        ]),

                    ], style={
                        "backgroundColor": "#1e1e2e",
                        "borderRadius": "16px",
                        "padding": "24px",
                        "border": "1px solid #2d2d44",
                        "boxShadow": "0 4px 20px rgba(0,0,0,0.3)"
                    })
                ], width=12)
            ], className="mt-4 mb-5"),

        ]),

    ], id="tabs", active_tab="tab-dashboard"),

], fluid=False, style={"backgroundColor": "#0d0d1a", "minHeight": "100vh", "padding": "20px"})


# ─── Helper: Empty Figure ─────────────────────────────────────────────────────
def empty_figure(msg="Loading..."):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#13131f",
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font=dict(size=16, color="#64748b"))],
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig


# ─── Helper: KPI Card ─────────────────────────────────────────────────────────
def kpi_card(title, value, color):
    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                html.P(title, style={"fontSize": "0.72rem", "textTransform": "uppercase",
                                     "letterSpacing": "1px", "color": "#64748b",
                                     "marginBottom": "8px"}),
                html.H4(value, style={"color": color, "fontWeight": "700",
                                      "margin": "0", "fontSize": "1.4rem"})
            ])
        ], style={"backgroundColor": "#1e1e2e", "border": "1px solid #2d2d44",
                  "borderRadius": "12px", "boxShadow": "0 4px 20px rgba(0,0,0,0.3)"}),
        xs=6, sm=6, md=3, lg=3
    )


# ─── Helper: Chat Bubble ──────────────────────────────────────────────────────
def chat_bubble(role, text):
    is_user = role == "user"
    return html.Div([
        html.Div(
            "You" if is_user else "🤖 AI Advisor",
            style={"fontSize": "0.72rem", "color": "#64748b", "marginBottom": "4px",
                   "textAlign": "right" if is_user else "left"}
        ),
        html.Div(
            html.Pre(text, style={
                "margin": "0", "fontSize": "0.88rem", "lineHeight": "1.6",
                "whiteSpace": "pre-wrap", "fontFamily": "Inter, sans-serif", "color": "white"
            }),
            style={
                "backgroundColor": "#7c3aed" if is_user else "#1a1a2e",
                "padding": "12px 16px",
                "borderRadius": "16px 16px 4px 16px" if is_user else "16px 16px 16px 4px",
                "maxWidth": "88%",
                "marginLeft": "auto" if is_user else "0",
                "border": "none" if is_user else "1px solid #2d2d44",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.2)"
            }
        )
    ], style={"display": "flex", "flexDirection": "column",
              "alignItems": "flex-end" if is_user else "flex-start"})


# ─── Helper: 200+ Stock Scanner ───────────────────────────────────────────────
def get_quick_market_data():
    us_large  = ["NVDA","AAPL","TSLA","MSFT","AMD","GOOGL","AMZN","META","NFLX","CRM"]
    us_mid    = ["PLTR","RKLB","IONQ","JOBY","ACHR","AEVA","LUNR","RXRX","SOUN","BBAI",
                 "NKLA","CLOV","WKHS","GOEV","KTOS","AVAV","SPCE","OPEN","HIMS","STEM"]
    us_small  = ["KULR","CRKN","PROP","HOLO","SRM","LIQT","MVST","ILLM","CDIO","NXTT",
                 "BFRI","PNTM","SHOT","FRZA","STTK","GRPN","CENN","MULN","ABSI","EVTL"]
    us_etf    = ["ARKK","ARKG","SOXL","TQQQ","FNGU"]

    india_large = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
                   "WIPRO.NS","TATAMOTORS.NS","ADANIPORTS.NS","BAJFINANCE.NS","SUNPHARMA.NS"]
    india_mid   = ["ZOMATO.NS","PAYTM.NS","NYKAA.NS","DELHIVERY.NS","POLICYBZR.NS",
                   "IRCTC.NS","TATAPOWER.NS","IDEA.NS","YESBANK.NS","RPOWER.NS",
                   "SUZLON.NS","NHPC.NS","IRFC.NS","RVNL.NS","RAILVIKAS.NS"]
    india_small = ["TRIDENT.NS","GEPIL.NS","INDIAMART.NS","EASEMYTRIP.NS","STOVEKRAFT.NS",
                   "FINEORG.NS","CENTURYPLY.NS","KPITTECH.NS","TANLA.NS","ROUTE.NS",
                   "CAMPUS.NS","LATENTVIEW.NS","HAPPYFORGE.NS","ORCHPHARMA.NS","BALRAMCHIN.NS"]

    all_tickers = (us_large + us_mid + us_small + us_etf +
                   india_large + india_mid + india_small)
    scored = []

    for t in all_tickers:
        try:
            df = yf.download(t, period="3mo", auto_adjust=True, progress=False)
            if df.empty or len(df) < 20:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            close  = df["Close"].squeeze().astype(float)
            volume = df["Volume"].squeeze().astype(float)

            price  = round(float(close.iloc[-1]), 2)
            chg_5d = round((close.iloc[-1] - close.iloc[-5])  / close.iloc[-5]  * 100, 2)
            chg_1m = round((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22] * 100, 2)

            delta     = close.diff()
            gain      = delta.clip(lower=0).rolling(14).mean()
            loss      = (-delta.clip(upper=0)).rolling(14).mean()
            rsi       = round(float((100 - (100 / (1 + gain / loss))).iloc[-1]), 1)
            vol_avg   = volume.rolling(20).mean().iloc[-1]
            vol_surge = round(float(volume.iloc[-1] / vol_avg), 2) if vol_avg > 0 else 1.0
            ma10      = close.rolling(10).mean().iloc[-1]
            ma30      = close.rolling(30).mean().iloc[-1]
            ma_signal = "BULLISH" if ma10 > ma30 else "BEARISH"

            score = 0
            if 30 < rsi < 65:   score += 3
            if chg_5d  > 2:     score += 2
            if chg_1m  > 5:     score += 2
            if vol_surge > 1.5: score += 2
            if ma_signal == "BULLISH": score += 1

            currency = "₹" if ".NS" in t else "$"
            cap_type = (
                "🔵 Large" if t in (us_large  + india_large) else
                "🟡 Mid"   if t in (us_mid    + india_mid)   else
                "🟢 Small" if t in (us_small  + india_small) else
                "📊 ETF"
            )
            scored.append({
                "ticker": t, "price": price, "currency": currency,
                "rsi": rsi, "chg_5d": chg_5d, "chg_1m": chg_1m,
                "vol_surge": vol_surge, "ma_signal": ma_signal,
                "score": score, "cap_type": cap_type,
            })
        except Exception:
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[:25]

    lines = ["=== TOP MOMENTUM PICKS (scored across 200+ stocks) ==="]
    for s in top:
        lines.append(
            f"{s['cap_type']} {s['ticker']}: Price={s['currency']}{s['price']}, "
            f"RSI={s['rsi']}, 5d={s['chg_5d']}%, 1m={s['chg_1m']}%, "
            f"VolSurge={s['vol_surge']}x, MA={s['ma_signal']}, Score={s['score']}/10"
        )
    lines.append("\n=== WEAK / AVOID (lowest momentum) ===")
    for s in scored[-5:]:
        lines.append(
            f"{s['ticker']}: RSI={s['rsi']}, 1m={s['chg_1m']}%, Score={s['score']}/10"
        )
    return "\n".join(lines)


# ─── Callback 1: Stock Search ─────────────────────────────────────────────────
@app.callback(
    Output("search-results-dropdown", "children"),
    Input("search-input", "value"),
    prevent_initial_call=True
)
def search_stocks(query):
    if not query or len(query) < 2:
        return []
    try:
        results = yf.Search(query, max_results=8).quotes
        if not results:
            return html.P("No results found.", style={"color": "#64748b", "fontSize": "0.82rem"})
        buttons = []
        for r in results:
            ticker   = r.get("symbol", "")
            name     = r.get("longname") or r.get("shortname") or ticker
            exchange = r.get("exchange", "")
            buttons.append(
                dbc.Button(f"{ticker} — {name} ({exchange})",
                    id={"type": "ticker-result", "index": ticker},
                    size="sm", color="dark", className="w-100 text-start mb-1",
                    style={"borderRadius": "6px", "fontSize": "0.82rem",
                           "backgroundColor": "#1e1e2e", "border": "1px solid #3d3d5c",
                           "color": "#e2e8f0", "padding": "7px 12px"})
            )
        return html.Div(buttons, style={
            "backgroundColor": "#13131f", "borderRadius": "8px",
            "padding": "8px", "border": "1px solid #2d2d44",
            "maxHeight": "220px", "overflowY": "auto"
        })
    except Exception as e:
        return html.P(f"Search error: {e}", style={"color": "#f87171", "fontSize": "0.82rem"})


# ─── Callback 2: Select Ticker ────────────────────────────────────────────────
@app.callback(
    Output("selected-ticker-store", "data"),
    Output("selected-ticker-display", "children"),
    Output("search-results-dropdown", "children", allow_duplicate=True),
    Input({"type": "ticker-result", "index": dash.ALL}, "n_clicks"),
    State({"type": "ticker-result", "index": dash.ALL}, "id"),
    prevent_initial_call=True
)
def select_ticker(n_clicks_list, id_list):
    if not any(n_clicks_list):
        return dash.no_update, dash.no_update, dash.no_update
    triggered = ctx.triggered_id
    if triggered:
        ticker = triggered["index"]
        return ticker, f"✅ {ticker}", []
    return dash.no_update, dash.no_update, dash.no_update


# ─── Callback 3: Run Model → All Charts ──────────────────────────────────────
@app.callback(
    Output("price-chart",             "figure"),
    Output("rsi-chart",               "figure"),
    Output("macd-chart",              "figure"),
    Output("feature-importance-chart","figure"),
    Output("signal-dist-chart",       "figure"),
    Output("signals-table",           "data"),
    Output("signals-table",           "columns"),
    Output("kpi-cards",               "children"),
    Output("error-alert",             "children"),
    Output("error-alert",             "style"),
    Input("run-btn",                  "n_clicks"),
    State("selected-ticker-store",    "data"),
    State("period-dropdown",          "value"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, ticker, period):
    ticker = ticker or "NVDA"
    try:
        raw_df    = fetch_data(ticker, period)
        feat_df   = engineer_features(raw_df)
        model, features, scaler, accuracy = train_model(feat_df)
        result_df = get_signals(feat_df, model, features, scaler)

        latest        = result_df.iloc[-1]
        current_price = round(float(latest["Close"]), 2)
        latest_signal = latest["Signal"]
        buy_pct       = round((result_df["Prediction"] == 1).mean() * 100, 1)

        base = dict(
            template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#13131f",
            font=dict(family="Inter, sans-serif", size=12, color="#e2e8f0"),
            autosize=True,
        )

        buys  = result_df[result_df["Prediction"] == 1]
        sells = result_df[result_df["Prediction"] == 0]

        # ── Candlestick + Bollinger Bands ──────────────────────────────
        price_fig = go.Figure()
        price_fig.add_trace(go.Candlestick(
            x=result_df.index,
            open=result_df["Open"].astype(float),
            high=result_df["High"].astype(float),
            low=result_df["Low"].astype(float),
            close=result_df["Close"].astype(float),
            name="Price",
            increasing_line_color="#4ade80",
            decreasing_line_color="#f87171"
        ))
        price_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["BB_Upper"].astype(float),
            name="BB Upper", line=dict(color="rgba(124,58,237,0.5)", width=1, dash="dot")
        ))
        price_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["BB_Lower"].astype(float),
            name="BB Lower", line=dict(color="rgba(124,58,237,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(124,58,237,0.05)"
        ))
        price_fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Close"].astype(float), mode="markers",
            marker=dict(symbol="triangle-up", size=12, color="#4ade80"), name="BUY"
        ))
        price_fig.add_trace(go.Scatter(
            x=sells.index, y=sells["Close"].astype(float), mode="markers",
            marker=dict(symbol="triangle-down", size=12, color="#f87171"), name="SELL"
        ))
        price_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["MA20"].astype(float),
            name="MA20", line=dict(color="#fb923c", width=1.5)
        ))
        price_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["MA50"].astype(float),
            name="MA50", line=dict(color="#22d3ee", width=1.5)
        ))
        price_fig.update_layout(
            **base,
            title=dict(text=f"{ticker} — Candlestick + Bollinger Bands + Signals",
                       font=dict(size=15)),
            xaxis_rangeslider_visible=False,
            yaxis=dict(tickformat=",.2f", automargin=True),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1),
            margin=dict(l=60, r=30, t=55, b=40), height=500
        )

        # ── RSI ────────────────────────────────────────────────────────
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["RSI"].astype(float),
            name="RSI", line=dict(color="#a78bfa", width=2),
            fill="tozeroy", fillcolor="rgba(124,58,237,0.08)"
        ))
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="#f87171",
            annotation_text="Overbought 70",
            annotation_font=dict(color="#f87171", size=11))
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="#4ade80",
            annotation_text="Oversold 30",
            annotation_font=dict(color="#4ade80", size=11))
        rsi_fig.update_layout(
            **base, title=dict(text="RSI Indicator", font=dict(size=15)),
            yaxis=dict(range=[0, 100], automargin=True),
            margin=dict(l=60, r=30, t=55, b=40), height=280
        )

        # ── MACD ───────────────────────────────────────────────────────
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["MACD"].astype(float),
            name="MACD", line=dict(color="#3b82f6", width=2)
        ))
        macd_fig.add_trace(go.Scatter(
            x=result_df.index, y=result_df["MACD_Signal"].astype(float),
            name="Signal Line", line=dict(color="#f97316", width=1.5)
        ))
        hist_colors = ["#4ade80" if v >= 0 else "#f87171"
                       for v in result_df["MACD_Hist"].astype(float)]
        macd_fig.add_trace(go.Bar(
            x=result_df.index, y=result_df["MACD_Hist"].astype(float),
            name="Histogram", marker_color=hist_colors, opacity=0.6
        ))
        macd_fig.update_layout(
            **base, title=dict(text="MACD Indicator", font=dict(size=15)),
            margin=dict(l=60, r=30, t=55, b=40), height=280, bargap=0
        )

        # ── Feature Importance ─────────────────────────────────────────
        importances = model.feature_importances_
        sorted_idx  = importances.argsort()
        fi_fig = go.Figure(go.Bar(
            x=importances[sorted_idx],
            y=[features[i] for i in sorted_idx],
            orientation="h",
            marker=dict(color=importances[sorted_idx],
                        colorscale="Purples", showscale=False),
            text=[f"{v:.3f}" for v in importances[sorted_idx]],
            textposition="outside"
        ))
        fi_fig.update_layout(
            **base, title=dict(text="Feature Importance", font=dict(size=15)),
            xaxis_title="Importance Score",
            margin=dict(l=120, r=60, t=55, b=40), height=400
        )

        # ── Signal Distribution ────────────────────────────────────────
        sig_counts = result_df["Signal"].value_counts()
        sd_fig = go.Figure(go.Pie(
            labels=sig_counts.index, values=sig_counts.values,
            marker=dict(colors=["#4ade80", "#f87171"]),
            hole=0.5, textinfo="label+percent",
            hoverinfo="label+value", textfont=dict(size=13)
        ))
        sd_fig.update_layout(
            **base, title=dict(text="Signal Distribution", font=dict(size=15)),
            margin=dict(l=40, r=40, t=55, b=40), height=400
        )

        # ── Signals Table ──────────────────────────────────────────────
        table_df = result_df[["Close","RSI","MA20","MA50","Signal"]].tail(10).reset_index()
        table_df.columns = ["Date","Close","RSI","MA20","MA50","Signal"]
        table_df["Date"] = table_df["Date"].astype(str).str[:10]
        for col in ["Close","RSI","MA20","MA50"]:
            table_df[col] = table_df[col].astype(float).round(2)

        columns = [{"name": c, "id": c} for c in table_df.columns]
        data    = table_df.to_dict("records")

        cards = [
            kpi_card("Current Price",  f"${current_price}", "#facc15"),
            kpi_card("Latest Signal",   latest_signal,       "#38bdf8"),
            kpi_card("Model Accuracy", f"{accuracy}%",       "#4ade80"),
            kpi_card("Buy Signals %",  f"{buy_pct}%",        "#a78bfa"),
        ]

        return (price_fig, rsi_fig, macd_fig, fi_fig, sd_fig,
                data, columns, cards, "", {"display": "none"})

    except Exception as e:
        traceback.print_exc()
        err_style = {
            "display": "block", "backgroundColor": "#450a0a",
            "color": "#f87171", "padding": "12px 20px",
            "borderRadius": "8px", "border": "1px solid #f87171",
            "marginBottom": "16px", "fontSize": "0.9rem"
        }
        empty = empty_figure("No data — check error above")
        return (empty, empty, empty, empty, empty,
                [], [], [], f"❌ Error: {str(e)}", err_style)


# ─── Callback 4: Download CSV ─────────────────────────────────────────────────
@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    State("signals-table", "data"),
    prevent_initial_call=True
)
def download_signals(n_clicks, table_data):
    if not table_data:
        return dash.no_update
    return dcc.send_data_frame(
        pd.DataFrame(table_data).to_csv, "trading_signals.csv", index=False
    )


# ─── Callback 5: Quick Prompts ────────────────────────────────────────────────
@app.callback(
    Output("chat-input", "value"),
    Input("quick-1", "n_clicks"),
    Input("quick-2", "n_clicks"),
    Input("quick-3", "n_clicks"),
    prevent_initial_call=True
)
def fill_quick_prompt(q1, q2, q3):
    prompts = {
        "quick-1": "I have ₹3000 and want to make it 10 times in 1 year. Which Indian stocks should I invest in?",
        "quick-2": "What are the best Indian stocks priced under ₹500 right now?",
        "quick-3": "What are the best US small cap stocks to buy right now?",
    }
    return prompts.get(ctx.triggered_id, "")


# ─── Callback 6: AI Chatbot ───────────────────────────────────────────────────
@app.callback(
    Output("chat-history", "children"),
    Output("chat-store",   "data"),
    Output("chat-input",   "value", allow_duplicate=True),
    Input("chat-send-btn", "n_clicks"),
    Input("chat-input",    "n_submit"),
    State("chat-input",    "value"),
    State("chat-store",    "data"),
    prevent_initial_call=True
)
def handle_chat(n_clicks, n_submit, user_message, chat_history):
    if not user_message or not user_message.strip():
        return dash.no_update, dash.no_update, dash.no_update

    chat_history = chat_history or []

    # Scan 200+ stocks (lightweight, no ML training)
    context_str = get_quick_market_data()

    # Detect currency
    is_inr = any(x in user_message.lower()
                 for x in ["₹", "rs", "rupee", "inr", "indian"])

    system_prompt = f"""You are a sharp AI quantitative trading advisor with access to a LIVE momentum scan of 200+ stocks including large cap, mid cap, small cap, and penny stocks from both US and Indian (NSE) markets.

LIVE SCAN RESULTS (scored by RSI, momentum, volume surge, MA crossover):
{context_str}

SCORING KEY: Score out of 10 — higher = more momentum signals firing simultaneously.
Cap types: 🔵 Large Cap | 🟡 Mid Cap | 🟢 Small Cap | 📊 ETF

CURRENCY DETECTED: {'Indian Rupees ₹ — prioritize .NS tickers from the scan' if is_inr else 'USD $ — prioritize US tickers from the scan'}

{'→ For Indian budget: only suggest stocks the user can AFFORD (price ≤ budget per share). Fractional investing is NOT available in India.' if is_inr else ''}

RULES:
1. Pull ALL recommendations FROM THE SCAN DATA above — never make up tickers
2. Prioritize HIGH SCORE stocks but also flag interesting small/mid caps with strong vol surge
3. For budget + goal questions: show a calculation table (required monthly %, realistic benchmark), then 3-5 specific picks from scan
4. For "best stocks" questions: ranked list with RSI/price/momentum reasoning per stock
5. For "is it possible?" questions: start with direct YES/NO verdict, then realistic scenarios
6. For 10x goals: focus on 🟡 Mid and 🟢 Small caps with score ≥ 5 and vol surge > 1.5x
7. Under 300 words — be direct and punchy
8. This is turn {len(chat_history)//2 + 1} — do NOT repeat anything from previous turns
9. Use **bold** for stock names and key numbers
10. End with: ⚠️ one-line risk note only — no paragraph disclaimers"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    try:
        print(f"[DEBUG] Groq call → {user_message[:60]}")
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.75,
            max_tokens=1024,
            top_p=0.9,
        )
        ai_reply = response.choices[0].message.content
        print(f"[DEBUG] Reply received ✓")
    except Exception as e:
        ai_reply = (f"⚠️ AI Error: {str(e)}\n\n"
                    "Check that your GROQ_API_KEY is set correctly in your .env or app.py.")

    chat_history.append({"role": "user",      "content": user_message})
    chat_history.append({"role": "assistant",  "content": ai_reply})

    bubbles = [chat_bubble(msg["role"], msg["content"]) for msg in chat_history]
    return bubbles, chat_history, ""


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)