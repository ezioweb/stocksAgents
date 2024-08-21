#Importação das Libs
import json
import os
from datetime import datetime
import yfinance as yf
from crewai import Agent, Task, Crew, Process
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st


# Criando Yahoo Finance Tool
def fetch_stock_price(ticket):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")
    return stock

yahoo_finance_tool = Tool(
    name = "Yahoo Finance Tool",
    description = "Fetches stock prices for {ticket} about a especific company",
    func= lambda ticket: fetch_stock_price(ticket)
)


# Importando OpenIA LLM - GPT
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo")


# In[5]:


#Criando o Agente Analista
stockPriceAnalyst = Agent(
    role="Senior stock price analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory="""You are a highly experienced in analyzing the price of an specific stock and make predictions about its future price.""",
    verbose=True,
    llm=llm,
    max_iter= 5,
    memory=True,
    allow_delegation=False,
    tools=[yahoo_finance_tool]
)


# In[6]:


get_stock_price = Task(
    description = "Analyze the stock {ticket} price history and create a trend analyses of up, down and sideways.",    
    expected_output = """Specify the current trend stock price - up, down or sideways.
     eg. stock='AAPL, price UP """,
    agent=stockPriceAnalyst
)


# In[7]:


# Importando a tool de search
search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)


# In[8]:


newsAnalyst = Agent(
    role="Stock news analyst",
    goal="""Create a short summary of the marcket news related to the stock {ticket} company. Specify the current trend 
     - up, down or sideways with the news context. For each request stock asset, specify a number between 0 and 100, 
     where 0 is extreme fear and 100 is extreme greed""",
    backstory="""You are a highly experienced in analyzing the market news and trends and have tracked assets for more 
    than 10 years.
    You are also master level analytics in the tradicional markets and have a deep understanding of human psychology.
    You understand news, theirs tittles and informations, but you look as those with a healthy dose of skeptism.
    You consider also the source of the news articles.""",
    verbose=True,
    llm=llm,
    max_iter= 10,
    memory=True,
    allow_delegation=False,
    tools=[search_tool]
)


# In[9]:


get_news = Task(
    description = f"""Take the stock and always include BTC to it (if not request).
    Use the search tool to search each one individually.
    The current date is {datetime.now()}.
    Composes the results into a helpfull report.
    """,    
    expected_output = """A summary of the overall market and one sentence summary for each request asset.
    Include a fear/greed score for each asset based on the news.
    Use format:
    <STOCK ASSET>
    <SUMMARY BASED ON NEWS>
    <TREND PREDICTION>
    <FEAR/GREED SCORE>""",
    agent=newsAnalyst
)


# In[10]:


stockAnalystWriter = Agent(
    role="Senior Stock Analyst Writer",
    goal="""Analyze the trend price and news to write a insightfull compelling and informative 4 paragraphs newsletter based in the stock report and price trend.""",
    backstory="""You are widely accepted as the best stock analyst in the market. 
    You understand complex concepts and create compelling stories and narratives that ressonate with wider audiences.
    You understand macro factors and combine theories - eg. cycle theory and fundamental analyses.
    You are able to hold multiple opinions when analyzing anything""",
    verbose=True,
    llm=llm,
    max_iter= 10,
    memory=True,
    allow_delegation=True    
)


# In[11]:


write_analyses = Task(
    description = """Use the stock price trend and the stock news to create an analyses and write the newsletter about
    the {ticket} company thats is brief and highlights the most important points.
    Focus on the stock price trend, news and fear/greed score. What are the near future considerations?
    Include the previous analyses of stock trend and news summary.
    """,    
    expected_output = """An eloquent 4 paragraphs newsletter formated as 'Markdown' in an easy readable manner.
    It should contain:
    # {ticket} Stock Analysys Newsletter 
    - 3 bullets executive summary;
    ## Introduction: 
    - set the overall picture and spike up the interest;
    ## Main Analysis:
    - Main part provides the meat of the analyses including the news summary and fear/greed score
    ## Summary:
    - Summary: key facts and concrete future trend prediction - up, down or sideways.  
    """,
    agent=stockAnalystWriter,
    context= [get_stock_price, get_news]
)


# In[12]:


crew = Crew(
    agents = [stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks = [get_stock_price, get_news],
    verbose = True,
    process= Process.hierarchical,    
    full_output=True,
    share_crew=False,
    manager_llm=llm,
    max_iter=15
)



# results=crew.kickoff(inputs={'ticket': 'AAPL'})


with st.sidebar:
    st.header('Enter the Stock code to research')
    with st.form(key='research_form'):
        topic = st.text_input("Slect the ticket")
        submit_button = st.form_submit_button(label="Run research")

if submit_button:
    if not topic:
        st.error("Please fill the ticket field" )
    else: 
        results=crew.kickoff(inputs={'ticket': topic})

        st.subheader("Results for your search:")
        st.write(results['final_output'])
