import streamlit as st
import pandas as pd
from streamlit.runtime.caching import cache_resource_api
import streamlit_nested_layout
import datetime
from classes import PairsHunter, PairsTrading
from streamlit_tags import st_tags
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings
from openai import OpenAI
import yfinance as yf

warnings.filterwarnings('ignore')

st.set_page_config(layout = 'wide')



@st.cache_resource
def get_openai_client():
    return OpenAI()

def get_openai_prompt_response(client, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            ia_role,
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

@st.cache_resource
def get_ai_response_async(prompt):
    client = get_openai_client()  
    response = get_openai_prompt_response(client, prompt)
    return response

@st.cache_resource
def download_data(stocks, inicio, fim):
    '''Download market data with Yahoo Finance, caching previously downloaded data'''
    cached_data = {}
    stockData = {}    
    stocks = [item.upper() for item in stocks]
    for stock in stocks:
        key = f"{stock.replace('.', '')}_{inicio}_{fim}"  
        if stock[-2:] == '.SA':
            ticker = yf.Ticker(stock)
            print(ticker.history(start=inicio, end=fim)['Close'])
        if key in cached_data:
            stockData[stock] = cached_data[key]
        else:
            try:
                ticker = yf.Ticker(stock)
                stockData[stock] = ticker.history(start=inicio, end=fim)['Close']
                cached_data[key] = stockData[stock]  

            except:
                st.error(f"Historical data not found for {stock}")
    
    df_stock = pd.DataFrame(stockData)
    df_stock.index = pd.to_datetime(df_stock.index)
    df_stock.index = df_stock.index.normalize()
    df_stock = df_stock.sort_index()
    df_stock = df_stock.groupby(level=0).apply(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
    df_stock = df_stock.droplevel(level=1)
    df_stock = df_stock.drop_duplicates()
    return df_stock




ia_role = {"role": "system", "content": "You are an analyst that have studied a lot about economics and geopolitical sciences"}

with st.sidebar:
    st.header("Pairs Strategy")

    data = pd.DataFrame()


    example_data = st.toggle(f"Use example dataset", value = True)
    if example_data:
        data = pd.read_excel('example.xlsx')
        data = data.set_index('Date')
        data = data.sort_values(by = 'Date')

    else:
        st.write(f"**Select tickers to understand correlation between them**")
        tickers = st_tags(
            label='_Tickers_ :dollar:',
            text='Press enter to add more',            
            suggestions=['VALE3.SA', 'KO', 'PEP', 'GOOG', 'MSTF', 'VALE', 'RIO', 'KCT', 'WPM'],    
            key='tickers')          

        inner_columns = st.columns(2)
        with inner_columns[0]:
            start = st.date_input(f"Start date :date:", value = datetime.date(2012, 1, 1), help = f"Start date of analysis :calendar:")

        with inner_columns[1]:
            end = st.date_input(f"End date :date:", value = datetime.date(2017, 1, 1), help = f"End date of analysis :calendar:")

        data = download_data(tickers, start, end)


    superior_filter = st.slider("Correlation filter", min_value = 0.25, max_value = 0.95,  value = 0.85, step = 0.05)
    
hunter_tab, strategy_tab = st.tabs(['Correlation Finder', 'Pairs Strategy'])
execution_ideas = []
cached_strategies = {}

with hunter_tab:
    hunter = PairsHunter(data, superior_filter)

    st.plotly_chart(hunter.plot_heatmap(), use_container_width=True)
    print(hunter.data)
    with st.expander("**Trading Ideas** :chart_with_upwards_trend:"):
        for idea in hunter.trading_ideas():
            inner_columns = st.columns([4, 6, 4])
            with inner_columns[0]:
                st.write(' x '.join(idea))
            
            with inner_columns[1]:
                add = st.toggle(f"Pairs Trading Analysis :heavy_plus_sign:", value = False, key ='add - x '.join(idea))
                if idea not in execution_ideas and add:
                    execution_ideas.append(idea)
                elif idea in execution_ideas and not add:
                    execution_ideas.remove(idea)
            
            with inner_columns[2]:
                ia_insight = st.toggle(f"AI Insight :robot_face: :sparkles:", value = False, key = 'ia - '.join(idea))
            
            if ia_insight:
                with st.expander(f" :clipboard: **Factors that may explain {' x '.join(idea)} correlation**"):
                    prompt = f"""Considering that I observe correlation between {idea[0]} (company or commodity stock) and {idea[1]} (company or commodity stock), which factors may explain? I don't need real-time data for my analysis"""

                    
                    ai_response = get_ai_response_async(prompt)  # Call asynchronously

                    openai_container = st.empty()
                    openai_container.write(ai_response) 

                    # with st.spinner("Waiting for AI response..."):
                    #     openai_response = get_openai_prompt_response(client, prompt)

                    # openai_container.write(openai_response)


with strategy_tab:
    # pair_strategy = PairsTrading()
    for idea in execution_ideas:
        with st.expander(f"**{' x '.join(idea)}**"):
            inner_columns = st.columns(2)
            with inner_columns[0]:
                k = st.slider("Standard deviation multiplier", value = 2.0, step = 0.25, min_value = 1.0, max_value = 5.0, key ='k - x '.join(idea))

            with inner_columns[1]:
                window_size = st.number_input("Window size", value = 20, min_value = 3, key ='ws - x '.join(idea))

            key = f"{'_'.join(idea)}_{k}_{window_size}" 
            if key not in cached_strategies:
                cached_strategies[key] = PairsTrading(data[idea], k, window_size)
            pair_strategy = cached_strategies[key]

            st.plotly_chart(pair_strategy.spread_chart_plotly())
            st.plotly_chart(pair_strategy.plot_backtest_plotly())

        

