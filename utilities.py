import pandas as pd
import numpy as np
import yfinance
import requests
import lxml
import pandas_datareader
from pandas_datareader import data as pdr
import yfinance as yf
from torch.nn.functional import softmax
from tqdm import tqdm

yf.pdr_override() # <== that's all it takes :-)

def financial_dataset(stock, num_of_labels=2, cutoff=0.25,
                      start_date="2010-01-01", end_date="2021-01-01") :
    ''' Downloads financial data for a stock and process it in the desired format
        Parameters :
          stock(str) : The desired stock's code
          cutoff(float) : A float indicating a percentage under which no price change is considered                                 increase or decrease eg. 0.25 = 0.25% price change from close-to-close
          num_of_labels(2 or 3) : Number of labels to use. 2 = [Increase,Decrease]. 
                                  3=[Increase, Decrease, Sideways]
          start_date(str) : "year-month-day" The day data collection will start .
          end_date(str) : "year-month-day" The day data collection will stop .    '''
    # parameter value check
    if (num_of_labels < 2 or num_of_labels > 3): 
        return print('Number of labels can be either 2 or 3')
                                                            
    fin_data = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
    
    print(f"{stock} financial dataframe dimensions ", fin_data.shape)
    
    # initialize price_change column 
    fin_data['Price_change'] = 1
    fin_data['date'] = 0
    dates = fin_data.index
    yesterday = str(dates[0].date())

    # How much should the price change in abs value to be considered increase/decrease.  
    for date in dates[1:] :
        today = str(date.date())

        yesterday_pr = fin_data.loc[yesterday, 'Close']
        today_pr = fin_data.loc[today, 'Close']
        diff = 100 * (today_pr - yesterday_pr)/yesterday_pr

        if (num_of_labels == 3) :
            if (diff > cutoff) :
                # price increase
                price_change = +1
            elif (diff < -cutoff) :
                # price decrease
                price_change = -1
            else:
                # almost steady price
                price_change = 0 
        elif (num_of_labels == 2 ): 
            if (diff > 0 ) : 
                # price increase
                price_change = +1
            elif (diff <= 0 ) :
                price_change = -1 
                                                                                                       
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

def read_news(stock):
    def read_rph(stock) :
        ''' Reads news relevant to 'stock' from the "raw_partner_headlines.csv" csv file. 
            Returns a dataframe in the format :[ Headline | date | stock  ] '''

        csv_path = 'Financial_News/raw_partner_headlines.csv'
        arp = pd.read_csv(csv_path)
        arp = arp.drop(columns=['Unnamed: 0', 'url', 'publisher'], axis=1)
        # Format the date column to match financial dataset
        arp['date'] = arp['date'].apply(lambda x: x.split(' ')[0] )
        news = arp[arp['stock'] == stock]
        print(f"The bot found {news.shape[0]} headlines from raw_partner_headlines.csv, regarding {stock} stock")
        return news

    def read_arp(stock) :
        ''' Reads news relevant to 'stock' from the "analyst_rating_processed.csv" csv file. 
        Returns a dataframe in the format :[ Headline | date | stock  ] '''
        csv_path = 'Financial_News/analyst_ratings_processed.csv'
        arp = pd.read_csv(csv_path)
        arp = arp.drop(columns=['Unnamed: 0'], axis=1)
        # pick the stock headlines
        arp = arp[arp['stock'] == stock]
        # Format the date column to match financial dataset (only keep date, not time)
        arp['date'] = arp['date'].apply(lambda x: str(x).split(' ')[0] )
        # Rename column title to headline to match other csv
        arp.rename({'title': 'headline'}, axis=1, inplace=True)
        news = arp
        print(f"The bot found {news.shape[0]} headlines from analyst_ratings_processed.csv, regarding {stock} stock")
        return news
    
    arp = read_arp(stock)
    rph = read_rph(stock)
    news = pd.concat([rph, arp], ignore_index=True)
    print(f"The bot found {news.shape[0]} headlines in total, regarding {stock} stock")
    return news
    

def merge_fin_news(df_fin, df_news, how='inner') :
    ''' Merges the financial data dataframe with the news dataframe and rearranges the column order
        how(str) : Merging technique : 'inner', 'outer' etc.. (check pd.merge documentation)      '''
    # merge on date column and only for their intersection
    merged_df = df_fin.merge(df_news, on='date', how=how)
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
    
    for i in tqdm(df.index) :
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

def merge_dates(df):
    '''
    Given a df that contains columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change],
    take the average of Positive, Negative, Neutral sentiment scores for each date and return a df that contains each
    date exactly one time. The return df has no column 'headline' since the scores now refer to an average of multiple
    news headlines.
        Parameters :
          df : A dataframe with columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change]

          returns df : aggragated sentiment scores by date with columns [date, stock, Open, Close, Volume, headline, Positive, Negative, Neutral, Price_change]
    '''

    # read the full enriched dataset in your main code like below and then pass it to the function
    # df = pd.read_csv('Financial_News/train_apple.csv', index_col=0, parse_dates=['date'])

    # take the average for Positive, Negative and Neutral columns by date. Drop headline column and all other columns per date are identical.
    dates_in_df = df['date'].unique()
    new_df = df.copy(deep=True).head(0)  # just take the df structure with no data inside
    new_df = new_df.drop(columns=['headline'])  # drop headline column

    for date in dates_in_df:
        sub_df = df[df['date'] == date]  # filter specific dates
        avg_positive = sub_df['Positive'].mean()
        avg_negative = sub_df['Negative'].mean()
        avg_neutral = sub_df['Neutral'].mean()
        sub_df = sub_df.drop(columns=['headline'])  # drop headline column

        stock = sub_df.iloc[0]['stock']
        open = sub_df.iloc[0]['Open']
        close = sub_df.iloc[0]['Close']
        volume = sub_df.iloc[0]['Volume']
        price_change = sub_df.iloc[0]['Price_change']

        sub_df = sub_df.head(0)  # empty sub_df to populate with just 1 row for each date
        # print(sub_df)
        sub_df.loc[0] = [date, stock, open, close, volume, avg_positive, avg_negative, avg_neutral,
                         price_change]  # populate the row
        # add sub_df's row to the new dataframe
        new_df = pd.concat([new_df, sub_df], axis=0, ignore_index=True)
    print(f" Dataframe now contains sentiment score for {new_df.shape[0]} different dates.")
    return(new_df)