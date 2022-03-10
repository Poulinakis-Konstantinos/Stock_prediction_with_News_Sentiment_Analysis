import pandas as pd
import numpy as np
import yfinance
import requests
import lxml
import pandas_datareader
from pandas_datareader import data as pdr
import yfinance as yf
from torch.nn.functional import softmax

yf.pdr_override() # <== that's all it takes :-)

def financial_dataset(stock, cutoff) :
    ''' Downloads financial data for a stock and process it in the desired format
        Parameters :
          stock(str) : The desired stock's code
          cutoff(float) : A float indicating a percentage under which no price change is considered increase or decrease eg. 0.25 = 0.25% price change from close-to-close'''
    
    fin_data = pdr.get_data_yahoo(stock, start="2010-01-01", end="2021-01-01")
    
    print(f"{stock} dataframe dimensions ", fin_data.shape)
    
    # initialize price_change column 
    fin_data['Price_change'] = 0
    fin_data['date'] = 0
    dates = fin_data.index
    yesterday = str(dates[0].date())

    # How much should the price change in abs value to be considered increase/decrease.  
    cutoff = 0.25
    for date in dates[1:] :
        today = str(date.date())

        yesterday_pr = fin_data.loc[yesterday, 'Close']
        today_pr = fin_data.loc[today, 'Close']
        diff = 100 * (today_pr - yesterday_pr)/yesterday_pr


        if (diff > cutoff) :
            # price increase
            price_change = +1
        elif (diff < -cutoff) :
            # price decrease
            price_change = -1
        else:
            # almost steady price
            price_change = 0 

        yesterday = today
        fin_data.loc[today,'Price_change'] = price_change
        fin_data.loc[today,'date'] = today

    incr = fin_data[fin_data['Price_change'] == 1 ].shape[0]
    decr = fin_data[fin_data['Price_change'] == -1 ].shape[0]
    stable = fin_data[fin_data['Price_change'] == 0 ].shape[0]
    print(f'Positive changes : {incr}')
    print(f'Negative changes : {decr}')
    print(f'No changes : {stable}')

    fin_data = fin_data.drop(columns = ['Low', 'High', 'Adj Close'], axis=1)
        
    return fin_data


def read_rph(stock) :
    ''' Reads news relevant to 'stock' from the "raw_partner_headlines.csv" csv file. 
        Returns a dataframe in the format :[ Headline | date | stock  ] '''
    
    csv_path = 'Financial_News/raw_partner_headlines.csv'
    arp = pd.read_csv(csv_path)
    arp = arp.drop(columns=['Unnamed: 0', 'url', 'publisher'], axis=1)
    # Format the date column to match financial dataset
    arp['date'] = arp['date'].apply(lambda x: x.split(' ')[0] )
    news = arp[arp['stock'] == stock]
    print(f"Read {news.shape[0]} headlines from raw_partner_headlines.csv, regarding {stock} stock")
    return news

def merge_fin_news(df_fin, df_news) :
    ''' Merges the financial data dataframe with the news dataframe and rearranges the column order '''
    # merge on date column and only for their intersection
    merged_df = df_fin.merge(df_news, on='date', how='inner')
    # rearrange column order
    merged_df = merged_df[['date', 'stock', 'Open', 'Close', 'Volume',  'headline', 'Price_change']]
    return merged_df
    


def sentim_analyzer(df, tokenizer, model):
    ''' Given a df that contains a column 'headline' with article healine texts, it runs inference on the healine with the 'model' (FinBert) 
       and inserts output sentiment features into the dataframe in the respective columns (Positive_sentim, Negative_sentim, Neutral_sentim)
       
        Parameters :
          df : A dataframe that contains headlines in a column called 'headline' . 
          tokenizer(AutoTokenizer object) : A pre-processing tokenizer object from Hugging Face lib. 
          model (AutoModelForSequenceClassification object) : A hugging face transformer model.     
          
          returns df : The initial dataframe with the 3 sentiment features as columns for each headline'''
    
    for i in df.index :
        try:
            headline = df.loc[i, 'headline']
        except:
            return print(' \'headline\' column might be missing from dataframe')
        # Pre-process input phrase
        input = tokenizer(headline, padding = True, truncation = True, return_tensors='pt')
        # Estimate output
        output = model(**input)
        # Pass model output logits through a softmax layer.
        predictions = softmax(output.logits, dim=-1)
        df.loc[i, 'Positive'] = predictions[0][0].tolist()
        df.loc[i, 'Negative'] = predictions[0][1].tolist()
        df.loc[i, 'Neutral']  = predictions[0][2].tolist()
    # rearrange column order
    try:
        df = df[['date', 'stock', 'Open', 'Close', 'Volume',  'headline', 'Positive', 'Negative', 'Neutral','Price_change']]
    except:
        pass
    return df