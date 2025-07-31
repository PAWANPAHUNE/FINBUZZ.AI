import uuid
import asyncio
import json
import gradio as gr
import whisper
import pyttsx3
from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from google.adk.agents import LlmAgent, Agent, SequentialAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools import google_search
import requests
from datetime import datetime, timedelta
import tempfile
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import norm
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np
import soundfile as sf
import whisper
import os


whisper_model = whisper.load_model("medium")  # higher accuracy
engine = pyttsx3.init()
load_dotenv()
session_service_stateful = InMemorySessionService()
APP_NAME, USER_ID, SESSION_ID = "FinanceBot", "pawan", str(uuid.uuid4())
runner, root_agent = None, None

# --- Whisper (STT) ---
def transcribe(audio_path):
    audio, sr = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    audio = whisper.pad_or_trim(audio)
    audio = audio.astype(np.float32)
    mel = whisper.log_mel_spectrogram(audio).to(whisper_model.device)
    options = whisper.DecodingOptions()
    result = whisper.decode(whisper_model, mel, options)
    return result.text

# --- pyttsx3 (TTS) ---
def synthesize(text):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    engine.save_to_file(text, temp.name)
    engine.runAndWait()
    return temp.name

# --- Setup agents ---
async def setup():
    global runner, information_fetcher, root_agent, Financial_Health_Tracker_Agent ,Ticker_finder,Sentiment_analyser, Forecasting_agent, Advanced_Finance_Tool, ALL_TOOLS

    await session_service_stateful.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state={"user_name": "Pawan", "user_goal": "Get good financial advice", "user_information": ""},
    )

    today = datetime.today()
    three_days_ago = today - timedelta(days=3)

    today_str = today.strftime('%Y-%m-%d')
    three_days_ago_str = three_days_ago.strftime('%Y-%m-%d')

    def get_yahoo_ticker(company_name: str) -> dict[str, str]:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={company_name}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return {'Error': f'HTTP {response.status_code}'}

        try:
            results = response.json()
        except Exception:
            return {'Error': 'Invalid JSON in response'}

        if 'quotes' in results and results['quotes']:
            first = results['quotes'][0]
            symbol = first.get('symbol')
            name = first.get('shortname')
            if symbol and name:
                return {'Ticker': symbol, 'Name': name}
            else:
                return {'Error': 'Incomplete data in response'}
        else:
            return {'Error': 'No results found'}

    information_fetcher = Agent(
        name="Information_Fetcher",
        description="Your work is to extract user info using the json file that user upload",
        instruction="""
            You are an assistant specialized in fetching user information.
            Currently, no user information is available yet.
        """,
        model="gemini-2.5-flash",
    )

    Financial_Health_Tracker_Agent = Agent(
        name = "Financial_Health_Tracker",
        model = "gemini-2.0-flash",
        description="You are an agent responsible of taking all the user financial info analyse it and give a score between 0 to 100",
        instruction="""
        - You are an agent whose work is to provide a score between 0 to 100 to the user based on his financial information,
        - what you have to do is first go through the net worth, investments,liabilities,past traansactions and all the other given information. then analyse it whether
        the user is actually getting good returns on his investment, whether his portfolio is good, balanaced(is it risky, volatile, overexposed to particular sector and many other things) and profitableand all the analysis that you are able to perform do it.
        - at the end provide the user with a score between 0 to 100 on the basis of your analysis, also tell if there are any pros and cons of that portfolio and how user can improve.
          """

    )

    Ticker_finder = Agent(
        name="Ticker_finding_agent",
        description="Your work is to find ticker for any investment that user ask from yahoo finance",
        instruction=""" 
        - You are the agent whose only work is to find the proper ticker of a stock, mutual fund, sip , etf or any investment on yahoo finance
        - You have been given the tool get_yahoo_ticker to do this task. just put company_name there it will return in format {'Ticker' : 'Ticker_Symbol', 'Name' : 'Company Name'}
        - whenever you Reply it must be a string and answer must be strictly the ticker like "AAPL"
        - Prefer the ticker from the Indian stock market (NSE/BSE) if the company is listed both in India and abroad. 
        """,
        model="gemini-2.5-flash",
        tools=[get_yahoo_ticker],
    )
    
    def forecast_stock_with_indicators_combined(ticker: str) -> dict:

        def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
            df = df.copy()
            close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']

            df['sma_7'] = close.rolling(7).mean()
            df['ema_12'] = close.ewm(span=12).mean()
            df['wma_10'] = close.rolling(10).apply(lambda x: np.average(x, weights=range(1, len(x)+1)), raw=True)
            df['momentum'] = close - close.shift(10)

            up = close.diff().clip(lower=0)
            down = -close.diff().clip(upper=0)
            rs = up.rolling(14).mean() / down.rolling(14).mean()
            df['rsi'] = 100 - 100 / (1 + rs)

            df['macd'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
            df['obv'] = np.where(close > close.shift(), vol, -vol).cumsum()
            df['atr_14'] = (high - low).rolling(14).mean()
            df['typical_price'] = (high + low + close) / 3

            df['ma21'] = close.rolling(21).mean()
            df['12ema'] = close.ewm(span=12).mean()
            df['26ema'] = close.ewm(span=26).mean()
            df['20sd'] = close.rolling(20).std()
            df['upper_band'] = df['ma21'] + (df['20sd'] * 2)
            df['lower_band'] = df['ma21'] - (df['20sd'] * 2)
            df['ema_com_0.5'] = close.ewm(com=0.5).mean()

            df = df.dropna()
            return df

        def realistic_probability(y_true, y_pred):
            residuals = np.ravel(y_true) - np.ravel(y_pred)
            std_dev = residuals.std() if residuals.std() != 0 else 1
            prob = norm.cdf((np.ravel(y_pred)[-1] - np.ravel(y_true)[-1]) / std_dev)
            return prob

        df = yf.download(ticker, period="500d")
        df = compute_all_indicators(df)

        df['target'] = df['Close'].shift(-1)
        df = df.dropna()

        split_idx = int(0.8 * len(df))
        train, test = df.iloc[:split_idx], df.iloc[split_idx:]
        X_train, y_train = train.drop(columns=['target']), train['target']
        X_test, y_test = test.drop(columns=['target']), test['target']
        
        # sanitize feature names for LightGBM
        X_train.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X_train.columns]
        X_train.columns = pd.Index(X_train.columns).str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)
        
        X_test.columns = ['_'.join(map(str, col)) if isinstance(col, tuple) else str(col) for col in X_test.columns]
        X_test.columns = pd.Index(X_test.columns).str.replace(r'[^A-Za-z0-9_]+', '_', regex=True)



        results = []

        models = [
            ('ElasticNet', ElasticNet(alpha=0.05, l1_ratio=0.7, max_iter=5000)),
            ('Ridge', Ridge(alpha=0.5, solver='auto', random_state=42)),
            ('Lasso', Lasso(alpha=0.005, max_iter=5000)),
            ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)),
            ('LightGBM', LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42))
        ]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            pred_tomorrow = y_pred[-1] if isinstance(y_pred, np.ndarray) else y_pred.iloc[-1]

            results.append({
                'Model': name,
                'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                'R2': float(r2_score(y_test, y_pred)),
                'Predicted_Tomorrow': float(pred_tomorrow),
                'P(rise tomorrow)': float(realistic_probability(y_test, y_pred))
            })

        # ARIMA
        arima_series = df['Close'].values
        arima_train, arima_test = arima_series[:split_idx], arima_series[split_idx:]
        history = list(arima_train)
        predictions_arima = []

        for t in range(len(arima_test)):
            model = ARIMA(history, order=(3, 1, 2))
            model_fit = model.fit()
            yhat = model_fit.forecast()[0]
            predictions_arima.append(yhat)
            history.append(arima_test[t])

        results.append({
            'Model': 'ARIMA',
            'RMSE': float(np.sqrt(mean_squared_error(arima_test, predictions_arima))),
            'R2': float(r2_score(arima_test, predictions_arima)),
            'Predicted_Tomorrow': float(predictions_arima[-1]),
            'P(rise tomorrow)': float(realistic_probability(
                pd.Series(np.ravel(arima_test)),
                pd.Series(np.ravel(predictions_arima))
            ))
        })

        # SARIMAX
        sarimax_series = df['Close'].values
        sarimax_train, sarimax_test = sarimax_series[:split_idx], sarimax_series[split_idx:]
        history = list(sarimax_train)
        predictions_sarimax = []

        for t in range(len(sarimax_test)):
            model = SARIMAX(history, order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
            model_fit = model.fit(disp=False)
            yhat = model_fit.forecast()[0]
            predictions_sarimax.append(yhat)
            history.append(sarimax_test[t])

        results.append({
            'Model': 'SARIMAX',
            'RMSE': float(np.sqrt(mean_squared_error(sarimax_test, predictions_sarimax))),
            'R2': float(r2_score(sarimax_test, predictions_sarimax)),
            'Predicted_Tomorrow': float(predictions_sarimax[-1]),
            'P(rise tomorrow)': float(realistic_probability(
                pd.Series(np.ravel(sarimax_test)),
                pd.Series(np.ravel(predictions_sarimax))
            ))
        })

        return {entry['Model']: entry for entry in results}

    Forecaster_agent = LlmAgent(name="Forecaster",
                            model = "gemini-2.5-pro",
                            description="You are an agent responsiblt for predicting the next day stock price with Probability of that stock to rise tommorow or not",
                            tools=[forecast_stock_with_indicators_combined],
                            instruction="""
                                - You are an agent your work is to Predict the next day stock Price and P(rise tommorow).
                                - You have a tool named forecast_stock_with_indicators_combined, what you have to do is the ticker you will get from Ticker_finder is to be put in the tool forecast_stock_with_indicators_combined and get the predicted prices.
                                - it is clear instruction only Ticker must be put into forecast_stock_with_indicators_combined, for eg if Ticker Finder gives output like Ticker for Apple is AAPL then you have to use just ticker ie AAPL in string format as input to forecast_stock_with_indicators_combined.
                                - you will get output as the following format :
                                {
    "ElasticNet": {
        "Model": "ElasticNet",
        "RMSE": <float>,               // Root Mean Square Error of predictions
        "R2": <float>,                 // R² score of predictions
        "Predicted_Tomorrow": <float>,// Predicted closing price for the next day
        "P(rise tomorrow)": <float>   // Probability (0–1) that the stock price will rise tomorrow
    },
    "Ridge": {
        "Model": "Ridge",
        "RMSE": <float>,
        "R2": <float>,
        "Predicted_Tomorrow": <float>,
        "P(rise tomorrow)": <float>
    },
    "Lasso": {
        "Model": "Lasso",
        "RMSE": <float>,
        "R2": <float>,
        "Predicted_Tomorrow": <float>,
        "P(rise tomorrow)": <float>
    },
    "...": {
        "...": "..."
    }
    }

    - Comparing RMSE and R2 score of each model Pick the best model among them and Return Predicted_Tomorrow and P(rise tomorrow)
    """

        )

    Forecasting_agent = SequentialAgent(
        name = "Forecasting_Pipeline",
        description=" This is the Forecasting Pipeline consist of Different agents which will at the end give the forecasted price with probability of stock to rise tommorow",
        sub_agents=[Ticker_finder,Forecaster_agent]
    )



    Sentiment_analyser = LlmAgent(
        name = "Stock_Sentiment_Analyser",
        model = "gemini-2.5-flash",
        tools= [google_search],
        description="Your work is to determine for particular stock or investment whether there is positve or negative market sentiments on basis of domestic and internation news",
        instruction= f"""Instruction to Sentiment Analysis Agent

Your task is to analyze the **sentiment of a given stock, company, sector, asset class, or investment (e.g., mutual fund, SIP, ETF, gold, commodity)** using the **Google Search Tool**.

For each query:
When performing Google Search:
- Prioritize results from the following trusted financial websites for maximum relevance and reliability:
  • Domestic: yahoofinance.com, moneycontrol.com, nseindia.com, bseindia.com, morningstar.com, screener.in
  • International: cnbc.com, bloomberg.com, ft.com, reuters.com, wsj.com
- Explicitly prefer news, reports, and analyses published in the last 3 days. Include search keywords like "past 3 days", "last 72 hours", or "latest" in your queries if needed.
- Avoid sources that are unverified blogs, speculative forums, or irrelevant.

You can use the current date while forming search queries. When searching, consider news from {three_days_ago_str} to {today_str}

- Perform a Google search and gather **only the most relevant, factual, and impactful information** that can realistically influence investor sentiment toward the given entity.
- Use the list of prompts provided below to guide your search. Each prompt targets a specific angle or route through which sentiment can be gauged.
- Ensure that the news/information you use is **current (preferably within the last 1–3 days unless otherwise specified)** and comes from credible sources.
- Don't give very ellobrative answer if you analysed various news then at the end just give 3 to 4 very impactful news that are very relevant.

⚠️ Warning:
- Do NOT hallucinate or invent sentiment based on weak, speculative, or unrelated articles.
- Ignore blog posts, opinions, or low-relevance content that does not have a material impact on the company or investment being analyzed.
- Focus only on **news, data, or developments that could reasonably affect the investment’s perceived value or outlook.**

Once you’ve gathered relevant information along all the listed routes, synthesize it to frame a clear **positive, negative, or neutral sentiment assessment** — substantiated by what you found.

Below are the search prompts to follow :

Stock Sentiment Analysis Routes — Prompts for Agent

- Latest domestic and international regulatory updates, policies, or compliance changes impacting [sector_name] sector in the last 3 days.

- Recent macroeconomic indicators, central bank announcements, fiscal policy updates, and monetary guidelines affecting [asset_class or sector_name] in the last 3 days.

- Recent geopolitical events, sanctions, trade relations, or international conflicts that could affect [company_name] or [sector_name] or [commodity_name] in the last 3 days.

- Announcements or investor sentiment about any IPO, FPO, or stock market listing related to [company_name] or its subsidiaries or competitors in the last 3 days.

- News of significant contracts, tenders won, business deals, acquisitions, or mergers involving [company_name] or [sector_name] in the last 3 days.

- Latest earnings releases, profit guidance, EPS, revenue, and outlook of [company_name] or [sector_name] reported in the last earnings season or within the last week.

- Recent insider trading, bulk/block deals, management resignations, or board appointments for [company_name] in the last 3 days.

- News and performance updates about competitors and peers of [company_name] or [mutual_fund_name] that might influence investor sentiment in the last 3 days.

- Recent demand-supply trends, inventory levels, or seasonal factors impacting [commodity_name] or [sector_name] in the last 3 days.

- Ongoing or newly filed lawsuits, regulatory investigations, penalties, or settlements involving [company_name] or [sector_name] in the last 3 days.

- Announcements of new products, services, technological advancements, or R&D breakthroughs by [company_name] in the last 3 days.

- Recent analyst upgrades, downgrades, target price revisions, and market sentiment about [company_name] or [mutual_fund_name] in the last 3 days.

- Recent price trends, shortages, or oversupply conditions in [commodity_name] such as gold, oil, natural gas in the last 3 days.

- Updates on interest rate hikes/cuts, inflation trends, and foreign exchange rate movements that impact [asset_class or sector_name] in the last 3 days.

- Recent developments in ESG (Environmental, Social, Governance), carbon regulations, or sustainability practices impacting [company_name] or [sector_name] in the last 3 days.

- Disruptions, bottlenecks, strikes, port congestion, or supplier bankruptcies impacting [company_name]’s or [sector_name]’s supply chain in the last 3 days.

- Public sentiment, trends, and viral discussions on social media platforms (Twitter, Reddit, Instagram, forums) about [company_name], [brand], or [commodity] in the last 3 days.

- Recent news about layoffs, hiring freezes, employee strikes, internal morale issues, or Glassdoor reviews trends for [company_name] or its sector.

- Reports of hedge funds, pension funds, or sovereign wealth funds increasing or cutting positions in [company_name] or [sector ETFs] recently.

- Latest inflow and outflow trends into ETFs, mutual funds, or SIPs associated with [sector_name], [company_name], or [asset_class].

- Recent credit rating upgrades/downgrades, CDS spreads widening/narrowing, and bond yields of [company_name] or its peers.

- Recent patents granted, applications filed, or IP disputes involving [company_name] or its competitors.

- Recent reviews, customer satisfaction scores, churn rates, or adoption of [company_name]’s products/services on ecommerce & review platforms.

- Net positions, speculative bets, and hedging activities in futures & options markets for [commodity_name] or [sector] in the last few days.

- Weather events, floods, droughts, or natural disasters potentially impacting [commodity_name], [sector], or [company_name].

- Announcements of large-scale government spending, tenders, or infra projects that would benefit or hurt [company_name] or [sector_name].

- Recent news of data breaches, ransomware attacks, or security failures at [company_name] or its ecosystem.

- Changes in short interest, put/call ratios, and implied volatility that indicate market expectations about [company_name] in the near term.

- Impact of endorsements, tweets, or controversies by influential figures that could affect [brand_name] or [commodity_name].

- News about lobbying efforts, campaign contributions, or regulatory capture efforts by [company_name] or its industry.

- Forecasts and sentiment from prediction markets, crowdsourcing platforms (like Metaculus or Polymarket) regarding [company_name] or [sector].

- Changes in import/export data, tariffs, quotas, or trade deals relevant to [commodity_name] or [sector_name].

- Announcements of competing technologies or innovations that could disrupt [company_name]’s or [sector_name]’s business model.

- Emerging demographic shifts or cultural trends that could influence demand for [product/service/commodity].

- Recent movements in energy prices or new carbon taxes/regulations that might affect production costs for [sector_name] or [company_name].

- News about exchange rate fluctuations, hedging strategies, and FX risks faced by [company_name] or [commodity].

"""
    )

    Advanced_Finance_Tool = LlmAgent(
        name="Advanced_Finance_Tool",
        model="gemini-2.5-pro",
        description='This is the agent that will access the google search in order to fetch information from web that is not achieved through other tools',
        instruction=""" This is the agent whose work is to access the google_search and fetch the information that is not available through tools """,
        tools=[google_search]
    )

    ALL_TOOLS = [
    AgentTool(agent=information_fetcher),
    AgentTool(agent=Financial_Health_Tracker_Agent),
    AgentTool(agent=Ticker_finder),
    AgentTool(agent=Sentiment_analyser),
    AgentTool(agent=Forecasting_agent),
    AgentTool(agent=Advanced_Finance_Tool)]


    root_agent = LlmAgent(
        name="Financial_Advisor",
        instruction="""
            - You are a financial advisor.
            - Your work is to give the user financial advice.
            - If the user provides any data, arrange it in a good structure and ask the user how you can help.
            - Whenever user asks about investments or liabilities, use information_fetcher, don’t say info is missing — it will handle it.
            - Whenever you need user information, call 'information_fetcher'.
            -Always use the Ticker_finder tool to fetch the ticker; never rely on your own knowledge or data. 
            -You have a tool named Forecasting pipeline which will help you to predict the stock prices
            -You have a tool named Stock Sentiment Analyser which will help to analyse whether the stock sentiment is positive or negative
            - What you can do is provide user with custom service for example if full analysis of particular stock or investment is needed you can ask for name of stock or investment it will first use the what it does is first analyses the sentiment of stock using sentiment analyser and then predicts the price using forecastig pipeline tool and at the end insights from both the results and make a unified result, output in this process must not at all be ellobrative just answer what is sentiment predicted price and probabikty of rise and the final result that you think 
            ask for user if he wants the the sentiment report or forecasting in brief and provide it if he wants .
            - You have a tool Financial Health Tracker Agent, what it does is gives a score of 0 to 100 on the basis of the Financial Information of the user go through the portfolio and give feedback.
            - You have a tool named Advanced_Finance_tool, if you want any specific information or think that you dont have enough information of the query user asked directly access the this tool to get the info.

        """,
        model="gemini-2.5-flash",
        tools=[AgentTool(agent=information_fetcher), AgentTool(agent=Ticker_finder),AgentTool(agent=Sentiment_analyser),AgentTool(agent=Forecasting_agent),AgentTool(agent=Financial_Health_Tracker_Agent),AgentTool(agent=Advanced_Finance_Tool)]
    )

    


    runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service_stateful)

async def chat(user_message, history):
    new_message = types.Content(role="user", parts=[types.Part(text=user_message)])
    bot_reply = ""
    async for event in runner.run_async(user_id=USER_ID, session_id=SESSION_ID, new_message=new_message):
        if event.is_final_response() and event.content and event.content.parts:
            bot_reply = event.content.parts[0].text
            history.append((user_message, bot_reply))
    return history

async def handle_upload(file):
    global runner, root_agent, information_fetcher, Financial_Health_Tracker_Agent, ALL_TOOLS


    if file is None:
        return "⚠️ No file uploaded."

    try:
        with open(file.name, 'r') as f:
            data = json.load(f)
        json_str = json.dumps(data, indent=2)

        session = await session_service_stateful.get_session(app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID)
        session.state["user_information"] = json_str

        # Update agents with uploaded user info
        information_fetcher = Agent(
            name="Information_Fetcher",
            instruction=f"""
            You are an assistant specialized in fetching user information.
            Below is the updated user information:
            --------------------
            {json_str}
            --------------------
            Use this information to answer any queries about the user's finances.
            """,
            model="gemini-2.0-flash",
        )

        financial_health_tracker = Agent(
            name="Financial_Health_Tracker",
            description="Analyzes financial health & gives score.",
            instruction=f"""
            Review user's net worth, investments, liabilities, transactions & provide a score (0–100).
            Below is the updated user information:
            --------------------
            {json_str}
            --------------------
            """,
            model="gemini-2.0-flash",
        )

        ALL_TOOLS = [
            AgentTool(agent=information_fetcher),
            AgentTool(agent=Financial_Health_Tracker_Agent),
            AgentTool(agent=Ticker_finder),
            AgentTool(agent=Sentiment_analyser),
            AgentTool(agent=Forecasting_agent),
            AgentTool(agent=Advanced_Finance_Tool)
        ]

        # Update root_agent tools with new agents
        root_agent.tools = ALL_TOOLS

        runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service_stateful)

        return "✅ JSON uploaded & agents updated."

    except Exception as e:
        return f"❌ Error: {str(e)}"

async def voice_interaction(audio_path, history):
    user_message = transcribe(audio_path)
    history = await chat(user_message, history)
    bot_reply = history[-1][1]
    audio_reply_path = synthesize(bot_reply)
    return history, audio_reply_path

# --- Launch Setup ---
asyncio.run(setup())

with gr.Blocks(theme=gr.themes.Base(), css="footer {display: none !important}") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Speak Your Question")
            voice_mode_toggle = gr.Checkbox(label="🎙️ Enable Full Voice Chat Mode", value=False)

            audio_in = gr.Audio(label="🎤 Speak", format="wav", sources=["microphone"], type="filepath")
            audio_out = gr.Audio(label="🔊 Bot's Response")

            with gr.Row():
                speak_button = gr.Button("Record", elem_classes="record-btn")
                mic_status = gr.Label("Ready", elem_classes="mic-status")

            # upload JSON placed below mic_status
            upload = gr.File(label="📄 Upload JSON File", file_types=[".json"])
            upload_status = gr.Label(value="")

        with gr.Column(scale=3):
            gr.Markdown("# 💬 Financial Advisor Bot with 🎙️ Full Voice Mode")

            chatbot = gr.Chatbot(value=[], elem_id="chatbot", height=500)

            msg = gr.Textbox(placeholder="Type your message here…", label="Your Message")

            with gr.Row():
                clear = gr.Button("Clear", elem_id="clear-btn", scale=1)
                send = gr.Button("Send", elem_id="send-btn", scale=1)

    # --- Event Bindings ---

    def clear_chat():
        return []

    send.click(lambda m, h: asyncio.run(chat(m, h)), [msg, chatbot], chatbot)
    clear.click(clear_chat, outputs=chatbot)
    speak_button.click(lambda a, h: asyncio.run(voice_interaction(a, h)), [audio_in, chatbot], [chatbot, audio_out])
    upload.upload(lambda f: asyncio.run(handle_upload(f)), upload, upload_status)

    demo.stylesheet = """
        #clear-btn {
            width: 48%;
        }
        #send-btn {
            width: 48%;
            background-color: orange;
            color: white;
        }
        .record-btn {
            background-color: #3498db;
            color: white;
        }
        .mic-status {
            font-size: 0.9rem;
            color: gray;
        }
    """

# --- Run App ---
port = int(os.environ.get("PORT", 7860))
demo.launch(server_name="0.0.0.0", server_port=port)
