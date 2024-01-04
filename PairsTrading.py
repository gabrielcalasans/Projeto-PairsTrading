import streamlit as st
import pandas as pd
import streamlit_nested_layout
import datetime
from classes import PairsHunter, PairsTrading
from streamlit_tags import st_tags
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')


st.set_page_config(layout = 'wide')

with st.sidebar:
    st.header("Pairs Strategy")

    example_data = st.toggle(f"Use example dataset", value = True)
    if example_data:
        data = pd.read_excel('example.xlsx')
        data = data.set_index('Date')
        data = data.sort_values(by = 'Date')

    else:
        st.write(f"List of assets ticker to understand correlation between them")
        tickers = st_tags(
            label='_Tickers_ :dollar:',
            text='Press enter to add more',            
            suggestions=['BRKM3.SA', 'VALE3.SA', 'KO', 'PEP', 'GOOG', 'MSTF'],    
            key='tickers')          

        inner_columns = st.columns(2)
        with inner_columns[0]:
            start = st.date_input(f"Start date :date:", value = datetime.date(2012, 1, 1), help = f"Start date of analysis :calendar:")

        with inner_columns[1]:
            end = st.date_input(f"End date :date:", value = datetime.date(2017, 1, 1), help = f"End date of analysis :calendar:")

    superior_filter = st.slider("Correlation filter", min_value = 0.25, max_value = 0.95,  value = 0.85, step = 0.05)
    
hunter_tab, strategy_tab = st.tabs(['Correlation Finder', 'Pairs Strategy'])
execution_ideas = []

with hunter_tab:
    hunter = PairsHunter(data, superior_filter)

    st.plotly_chart(hunter.plot_heatmap(), use_container_width=True)

    with st.expander("Trading Ideas"):
        for idea in hunter.trading_ideas():
            inner_columns = st.columns([4, 6, 4])
            with inner_columns[0]:
                st.write(' x '.join(idea))
            
            with inner_columns[1]:
                add = st.toggle(f"Add to trading review", value = False, key ='add - x '.join(idea))
                if idea not in execution_ideas and add:
                    execution_ideas.append(idea)
                elif idea in execution_ideas and not add:
                    execution_ideas.remove(idea)

with strategy_tab:
    # pair_strategy = PairsTrading()
    for idea in execution_ideas:
        with st.expander(' x '.join(idea)):
            inner_columns = st.columns(2)
            with inner_columns[0]:
                k = st.slider("Standard deviation multiplier", value = 2.0, step = 0.25, min_value = 1.0, max_value = 5.0, key ='k - x '.join(idea))

            with inner_columns[1]:
                window_size = st.number_input("Window size", value = 20, min_value = 3, key ='ws - x '.join(idea))

            print(idea, k , window_size)
            pair_strategy = PairsTrading(data[idea], k, window_size)

            st.plotly_chart(pair_strategy.spread_chart_plotly())

            st.plotly_chart(pair_strategy.plot_backtest_plotly())

        

