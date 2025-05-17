## AI-Powered Technical Analysis Dashboard (Gemini 2.0)

# Libraries
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import google.generativeai as genai
import tempfile
import os
import json
import numpy as np
from datetime import datetime, timedelta

# Configure the API key - IMPORTANT: Use Streamlit secrets or environment variables for security
# For now, using hardcoded API key - REPLACE WITH YOUR ACTUAL API KEY SECURELY
# GOOGLE_API_KEY = st.secrets["MY_GOOGLE_API_KEY"]
GOOGLE_API_KEY = os.getenv("MY_GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Select the Gemini model - using 'gemini-2.0-flash' as a general-purpose model
MODEL_NAME = 'gemini-2.0-flash' # or other model
gen_model = genai.GenerativeModel(MODEL_NAME)

# Set up Streamlit app
st.set_page_config(layout="wide")
st.title("Analisi Tecnica AI")
st.sidebar.header("Configurazione")

# Input for multiple stock tickers (comma-separated)
tickers_input = st.sidebar.text_input("Inserisci il Ticker del Titolo (separati da virgola):", "AAPL,MSFT,ISP.MI")
# Parse tickers by stripping extra whitespace and splitting on commas
tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]

# Set the date range: start date = one year before today, end date = today
end_date_default = datetime.today()
start_date_default = end_date_default - timedelta(days=365)
start_date = st.sidebar.date_input("Data Inizio", value=start_date_default)
end_date = st.sidebar.date_input("Data Fine", value=end_date_default)

# Technical indicators selection (applied to every ticker)
st.sidebar.subheader("Indicatori Tecnici")
indicators = st.sidebar.multiselect(
    "Seleziona Indicatori:",
    ["20-Day SMA", "40-Day SMA", "20-Day EMA", "200-Day SMA", "20-Day Bollinger Bands", "VWAP", "Composite Momentum", "Heisenberg"],
    default=["20-Day SMA", "40-Day SMA", "200-Day SMA"]
)

# Button to fetch data for all tickers
if st.sidebar.button("Mostra Dati"):
    stock_data = {}
    for ticker in tickers:
        # Download data for each ticker using yfinance
        data = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
        if not data.empty:
            stock_data[ticker] = data
        else:
            st.warning(f"Nessun dato trovato per {ticker}.")
    st.session_state["stock_data"] = stock_data
    st.success("Dati Titoli caricati con successo per: " + ", ".join(stock_data.keys()))

# Ensure we have data to analyze
if "stock_data" in st.session_state and st.session_state["stock_data"]:

    # Define a function to calculate Composite Momentum
    def calculate_composite_momentum(data, periods=[5, 10, 20, 50]):
        # Calculate momentum for each period
        momentum = pd.DataFrame(index=data.index)
        
        for period in periods:
            # Momentum is calculated as percentage change over the period
            momentum[f'mom_{period}'] = data['Close'].pct_change(period) * 100
        
        # Composite momentum is the average of all period momentums
        momentum['composite'] = momentum.mean(axis=1)
        
        return momentum['composite']
    
    # Define a function to calculate Heisenberg Indicator
    def calculate_heisenberg(data, short_period=10, long_period=30):
        # Calculate short-term volatility
        short_vol = data['Close'].pct_change().rolling(window=short_period).std() * 100
        
        # Calculate long-term volatility
        long_vol = data['Close'].pct_change().rolling(window=long_period).std() * 100
        
        # Heisenberg indicator is the ratio of short-term to long-term volatility
        heisenberg = short_vol / long_vol
        
        return heisenberg

    # Define a function to build chart, call the Gemini API and return structured result
    def analyze_ticker(ticker, data):
        # Check if we need to display the subplots for momentum or heisenberg
        need_subplots = "Composite Momentum" in indicators or "Heisenberg" in indicators
        
        # Create figure with subplots if needed
        if need_subplots:
            # Count how many subplots we need (main chart + indicators)
            subplot_count = 1  # Main chart
            if "Composite Momentum" in indicators:
                subplot_count += 1
            if "Heisenberg" in indicators:
                subplot_count += 1
                
            # Create subplot grid
            specs = [[{"type": "candlestick"}]]
            for _ in range(subplot_count - 1):
                specs.append([{"type": "scatter"}])
                
            # Calculate row heights to make main chart larger
            # MODIFICATO: Aumentata l'altezza del grafico principale dall'originale 0.7 (70%) a 0.8 (80%)
            row_heights = [0.8]  # Main chart takes 80% of height
            remaining_height = 0.2
            for _ in range(subplot_count - 1):
                row_heights.append(remaining_height / (subplot_count - 1))
                
            # Crea la figura con i subplots e le altezze personalizzate
            fig = make_subplots(rows=subplot_count, cols=1, shared_xaxes=True, 
                               vertical_spacing=0.05, specs=specs, row_heights=row_heights)
                
            # Impostiamo l'altezza e larghezza totale
            fig.update_layout(height=800, width=1000)
        else:
            # Just create a regular figure
            fig = go.Figure()
            fig.update_layout(height=800, width=1000)
        
        # Add candlestick chart
        candlestick = go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name="Candlestick"
        )
        
        if need_subplots:
            fig.add_trace(candlestick, row=1, col=1)
        else:
            fig.add_trace(candlestick)

        # Add selected technical indicators
        def add_indicator(indicator):
            if indicator == "20-Day SMA":
                sma = data['Close'].rolling(window=20).mean()
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=sma, mode='lines', name='SMA (20)'))
            
            elif indicator == "40-Day SMA":
                sma40 = data['Close'].rolling(window=40).mean()
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=sma40, mode='lines', name='SMA (40)', 
                                            line=dict(color='orange', width=1.5)), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=sma40, mode='lines', name='SMA (40)', 
                                            line=dict(color='orange', width=1.5)))
            
            elif indicator == "200-Day SMA":
                sma200 = data['Close'].rolling(window=200).mean()
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=sma200, mode='lines', name='SMA (200)', 
                                            line=dict(color='purple', width=1.5)), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=sma200, mode='lines', name='SMA (200)', 
                                            line=dict(color='purple', width=1.5)))
            
            elif indicator == "20-Day EMA":
                ema = data['Close'].ewm(span=20).mean()
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=ema, mode='lines', name='EMA (20)'))
            
            elif indicator == "20-Day Bollinger Bands":
                sma = data['Close'].rolling(window=20).mean()
                std = data['Close'].rolling(window=20).std()
                bb_upper = sma + 2 * std
                bb_lower = sma - 2 * std
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=bb_upper, mode='lines', name='BB Upper'))
                    fig.add_trace(go.Scatter(x=data.index, y=bb_lower, mode='lines', name='BB Lower'))
            
            elif indicator == "VWAP":
                data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()
                if need_subplots:
                    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'), row=1, col=1)
                else:
                    fig.add_trace(go.Scatter(x=data.index, y=data['VWAP'], mode='lines', name='VWAP'))
            
            elif indicator == "Composite Momentum":
                # Calculate Composite Momentum
                momentum = calculate_composite_momentum(data)
                
                # Find the current subplot row
                row = 2
                if need_subplots:
                    # Add the momentum line
                    fig.add_trace(go.Scatter(x=data.index, y=momentum, mode='lines', 
                                            name='Composite Momentum', line=dict(color='green')), row=row, col=1)
                    
                    # Add media mobile a 7 periodi
                    momentum_ma7 = momentum.rolling(window=7).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=momentum_ma7, mode='lines', 
                                            name='Mom MA (7)', line=dict(color='blue', width=1.5, dash='dot')), 
                                row=row, col=1)
                    
                    # Add media mobile a 30 periodi
                    momentum_ma30 = momentum.rolling(window=30).mean()
                    fig.add_trace(go.Scatter(x=data.index, y=momentum_ma30, mode='lines', 
                                            name='Mom MA (30)', line=dict(color='red', width=1.5, dash='dot')), 
                                row=row, col=1)
                    
                    # Add a zero line for reference
                    fig.add_trace(go.Scatter(x=data.index, y=[0] * len(data.index), mode='lines', 
                                            name='Zero Line', line=dict(color='black', dash='dash')), row=row, col=1)
                    
                    # Update the y-axis title
                    fig.update_yaxes(title_text="Momentum %", row=row, col=1)
            
            elif indicator == "Heisenberg":
                # Calculate Heisenberg indicator
                heisenberg = calculate_heisenberg(data)
                
                # Find the current subplot row
                row = 2
                if "Composite Momentum" in indicators:
                    row = 3
                
                if need_subplots:
                    # Add the Heisenberg line
                    fig.add_trace(go.Scatter(x=data.index, y=heisenberg, mode='lines', 
                                            name='Heisenberg', line=dict(color='red')), row=row, col=1)
                    
                    # Add a reference line at 1.0
                    fig.add_trace(go.Scatter(x=data.index, y=[1] * len(data.index), mode='lines', 
                                            name='Reference', line=dict(color='black', dash='dash')), row=row, col=1)
                    
                    # Update the y-axis title
                    fig.update_yaxes(title_text="Heisenberg", row=row, col=1)

        # Add all selected indicators
        for ind in indicators:
            add_indicator(ind)
        
        # Update layout
        fig.update_layout(xaxis_rangeslider_visible=False)

        # Save chart as temporary PNG file and read image bytes
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
            fig.write_image(tmpfile.name)
            tmpfile_path = tmpfile.name
        with open(tmpfile_path, "rb") as f:
            image_bytes = f.read()
        os.remove(tmpfile_path)

        # Create an image Part
        image_part = {
            "data": image_bytes,  
            "mime_type": "image/png"
        }

        # Analysis prompt in Italian
        analysis_prompt = (
            f"Sei un Trader Specializzato in Analisi Tecnica presso una delle principali istituzioni finanziarie. "
            f"Analizza il grafico azionario per {ticker} basandoti sul grafico a candele e sugli indicatori tecnici visualizzati. "
            f"Fornisci una giustificazione dettagliata della tua analisi, spiegando quali pattern, segnali e trend osservi. "
            f"Poi, basandoti esclusivamente sul grafico, fornisci una raccomandazione tra le seguenti opzioni: "
            f"'Strong Buy' (Acquisto Forte), 'Buy' (Acquisto), 'Weak Buy' (Acquisto Debole), 'Hold' (Mantieni), 'Weak Sell' (Vendita Debole), 'Sell' (Vendita) o 'Strong Sell' (Vendita Forte). "
            f"Restituisci il risultato come oggetto JSON con due chiavi: 'azione' e 'giustificazione'."
        )

        # Call the Gemini API with text and image input - Roles added: "user" for both text and image
        contents = [
            {"role": "user", "parts": [analysis_prompt]},  # Text prompt with role "user"
            {"role": "user", "parts": [image_part]}       # Image part with role "user"
        ]

        response = gen_model.generate_content(
            contents=contents  # Pass the restructured 'contents' with roles
        )

        try:
            # Attempt to parse JSON from the response text
            result_text = response.text
            # Find the start and end of the JSON object within the text (if Gemini includes extra text)
            json_start_index = result_text.find('{')
            json_end_index = result_text.rfind('}') + 1  # +1 to include the closing brace
            if json_start_index != -1 and json_end_index > json_start_index:
                json_string = result_text[json_start_index:json_end_index]
                result = json.loads(json_string)
            else:
                raise ValueError("No valid JSON object found in the response")

        except json.JSONDecodeError as e:
            result = {"azione": "Error", "giustificazione": f"JSON Parsing error: {e}. Raw response text: {response.text}"}
        except ValueError as ve:
            result = {"azione": "Error", "giustificazione": f"Value Error: {ve}. Raw response text: {response.text}"}
        except Exception as e:
            result = {"azione": "Error", "giustificazione": f"General Error: {e}. Raw response text: {response.text}"}

        return fig, result

    # Create tabs: first tab for overall summary, subsequent tabs per ticker
    tab_names = ["Sommario"] + list(st.session_state["stock_data"].keys())
    tabs = st.tabs(tab_names)

    # List to store overall results
    overall_results = []

    # Process each ticker and populate results
    for i, ticker in enumerate(st.session_state["stock_data"]):
        data = st.session_state["stock_data"][ticker]
        # Analyze ticker: get chart figure and structured output result
        fig, result = analyze_ticker(ticker, data)
        overall_results.append({"Titolo": ticker, "Raccomandazione": result.get("azione", "N/A")})
        # In each ticker-specific tab, display the chart and detailed justification
        with tabs[i + 1]:
            st.subheader(f"Analisi per {ticker}")
            st.plotly_chart(fig)
            st.write("**Giustificazione Dettagliata:**")
            st.write(result.get("giustificazione", "Nessuna giustificazione."))

    # In the Overall Summary tab, display a table of all results
    with tabs[0]:
        st.subheader("Raccomandazioni Generali")
        df_summary = pd.DataFrame(overall_results)
        st.table(df_summary)
else:
    st.info("Inserisci i dati nella barra laterale.")
