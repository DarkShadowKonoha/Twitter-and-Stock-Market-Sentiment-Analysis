#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
import seaborn as sns
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import dask.dataframe as dd


# In[2]:


# nltk.download('vader_lexicon')


# In[2]:


prices = pd.read_csv('values-of-top-nasdaq-copanies-from-2010-to-2020/CompanyValues.csv')


# In[4]:


# prices.head()


# In[3]:


prices['day_date'] = pd.to_datetime(prices['day_date'], format="%Y-%m-%d").dt.date.astype('datetime64[ns]')


# In[4]:


prices = prices.sort_values(by=['day_date']).reset_index()


# In[5]:


prices = prices.rename(columns={"day_date":"date"})


# In[6]:


def create_indicator(data):
    prices = data.sort_values(by=['date']).reset_index()

    # Creating Simple Moving Average
    n = [10, 20, 50, 100]
    for i in n:
        prices.loc[:,(str("MA"+str(i)))]=prices['close_value'].rolling(i).mean()

    # Calculate MACD
    day26 = prices['close_value'].ewm(span=26, adjust=False).mean()
    day12 = prices['close_value'].ewm(span=12, adjust=False).mean()
    prices.loc[:,('macd')] = day12-day26
    prices.loc[:,('signal')]=prices['macd'].ewm(span=9, adjust=False).mean()

    # Calculate RSI
    up = np.log(prices.close_value).diff(1)
    down = np.log(prices.close_value).diff(1)

    up[up<0]=0
    down[down>0]=0

    roll_up = up.ewm(span=14).mean()
    roll_down = down.abs().ewm(span=14).mean()

    RS1 = roll_up / roll_down
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    prices.loc[:,('rsi')]=RSI1

    return prices

d = dict(tuple(prices.groupby('ticker_symbol')))
d = {k:create_indicator(v) for k, v in d.items()}

def subset_prices(d, ticker, start, end):
    x=d[ticker]
    x=x[((x.date>=start)&(x.date<=end))]
    return x


# In[7]:


tweets = pd.read_csv('tweets-about-the-top-companies-from-2015-to-2020/Tweet.csv')
company_tweet = pd.read_csv('tweets-about-the-top-companies-from-2015-to-2020/Company_Tweet.csv')


# In[8]:


tweets=tweets.merge(company_tweet,how='left',on='tweet_id')


# In[9]:


# tweets.shape


# In[12]:


# tweets.head()


# In[10]:


tweets['date'] = pd.to_datetime(tweets['post_date'], unit='s').dt.date
tweets.date = pd.to_datetime(tweets.date,errors='coerce')
tweets['time'] = pd.to_datetime(tweets['post_date'], unit='s').dt.time


# In[11]:


# tweets['ticker_symbol'].unique()


# In[12]:


sia = SentimentIntensityAnalyzer()


# In[13]:


# def get_sentiment(tweets,ticker='TSLA',start='2017-01-01',end='2017-02-01'):
#     #sbuset
#     df=tweets.loc[((tweets.ticker_symbol==ticker)&(tweets.date>=start)&(tweets.date<=end))]
#     # apply the SentimentIntensityAnalyzer
#     df.loc[:,('score')]=df.loc[:,'body'].apply(lambda x: sia.polarity_scores(x)['compound'])
#     # create label
#     #bins= pd.interval_range(start=-1, freq=3, end=1)
#     df.loc[:,('label')]=pd.cut(np.array(df.loc[:,'score']),bins=[-1, -0.66, 0.32, 1],right=True ,labels=["bad", "neutral", "good"])
    
#     df=df.loc[:,["date","score","label","tweet_id","body"]]
#     return df


# In[17]:


# print('tesla misses earnings, analyst suggest downgrade, sell now ')
# sia.polarity_scores('tesla misses earnings, analyst suggest downgrade , sell now ')


# In[14]:


# augment vocab to include words related to stock market

positive_words='buy bull long support undervalued underpriced cheap upward rising trend moon rocket hold breakout call beat support buying holding high profit'
negative_words='sell bear bubble bearish short overvalued overbought overpriced expensive downward falling sold sell low put miss resistance squeeze cover seller '

dictOfpos = {i : 4 for i in positive_words.split(" ")}
dictOfneg = {i : -4 for i in negative_words.split(" ")}
Financial_Lexicon = {**dictOfpos, **dictOfneg}

sia.lexicon.update(Financial_Lexicon)

# print('tesla misses earnings, analyst suggest downgrade, sell now ')
# sia.polarity_scores('tesla misses earnings, analyst suggest downgrade , sell now ')


# In[15]:


def price_plot_ma(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr

    n = df.shape[0] # number of dates
    if ax is None:
        ax = plt.gca()
        
    # format data for seaborn
    df=df.melt(id_vars='date',var_name='var', value_name='vals')
    df=df[df['var'].isin(['close_value','MA10','MA20','MA50','MA100'])]
    df['vals']=df['vals'].astype(float)
    df.index=df.date.dt.date
    df.date=df.date.dt.date
    # set axis formats / Set the locator
    if ax is None:
        ax = plt.gca()
        
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    

        
    ax.set_ylabel('Close Price')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    sns.lineplot(data=df, x='date', y='vals',hue='var',palette='cool_r',ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    return ax


# In[16]:


def price_plot_vol(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    
    n=df.shape[0]
    
    df.index=df.date.dt.date
    if ax is None:
        ax = plt.gca()
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    
    ax.set_ylabel('Traded Volume (million)')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x/1000000), ',')))
    sns.lineplot(data=df, x='date', y='volume',palette='cool_r',ax=ax)

    return ax


# In[17]:


def sentiment_barplot(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    
    df=df.groupby(['date','label'])['tweet_id'].agg('count').reset_index(name="count")
    
    n=len(df.date.unique())
    
    # format the data and make proportion
    df=df.pivot(index='date',columns='label',values='count')
    df=pd.DataFrame(df.to_records()).reset_index()
    df.loc[:,"total"]=df.loc[:,['bad','neutral','good']].sum(axis=1)
    df.loc[:,['bad','neutral','good']]=df.loc[:,['bad','neutral','good']].div(df.total,axis=0)
    df.loc[:,"total"]=df.loc[:,['bad','neutral','good']].sum(axis=1)
    df=df.drop(['total'], axis=1)
   
    df.index=df.date.dt.date
    if ax is None:
        ax = plt.gca()
    colors=['crimson','lightgrey','mediumseagreen']
    df.loc[:,['bad','neutral', 'good']].plot.bar(stacked=True, color=colors, width=1.0,alpha=0.5,ax=ax)
    
   
    # set axis formats / Set the locato
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
         
    
    ax.set_ylabel('Sentiment')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    
    ax.grid(True, which='major',axis='both')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    return ax


# In[18]:


def sentiment_tweet_vol(df,ax=None,**kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    df=df.groupby(['date'])['label'].agg('count').reset_index(name="count")
    df.index=df.date.dt.date
    n=len(df.date.unique())
    
    if ax is None:
        ax = plt.gca()
    # set axis formats / Set the locator
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    ax.set_ylabel('Tweet Volume')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    sns.lineplot(data=df, x='date', y='count',palette='cool_r',ax=ax)
    
    return ax


# In[19]:


def corr_plot(sp,tw):
    
    x=tw.groupby(['date','label']).agg({"score":['count','mean']}).unstack('label') 
    sp=sp.reset_index(drop=True)
    # format the data and make proportion
    x=pd.DataFrame(x.to_records())
    # format columns names
    x.columns=['date','count_bad','count_neutral','count_good','score_mean_bad','score_mean_neutral','score_mean_good']
    x.loc[:,'tweet_volume']=x.loc[:,['count_bad','count_neutral','count_good']].sum(axis=1)
    x.loc[:,'count_ratio_gb']=x.count_good/x.count_bad # create a ratio good:bad
    # join price
    x=x.merge(sp.loc[:,['date','MA10', 'MA20', 'MA50','MA100', 'macd', 'rsi','volume']],how='left',left_on='date',right_on='date')

    corr = x.corr()
    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(corr)
    ax = sns.heatmap(
        round(corr,3),
        vmin=-1, vmax=1, center=0,
        cmap="YlGnBu",annot=True,annot_kws={"fontsize":8}, fmt=".2",
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    return ax


# In[24]:


# start='2015-01-01'
# end='2020-12-31'
# ticker='TSLA'
# # get data
# sp=subset_prices(d,ticker,start,end) #get price info
# fig,ax=plt.subplots(figsize=(12, 8))
# fig.suptitle(ticker+ ": Price,Moving Averages",fontsize=14,horizontalalignment='right', verticalalignment='top')
# price_plot_ma(ax=ax,df=sp)


# In[25]:


# start='2018-06-01'
# end='2019-12-31'
# ticker='TSLA'
# # get data
# sp=subset_prices(d,ticker,start,end) #get price info
# tw=get_sentiment(tweets,ticker,start,end) # get tweets


# In[26]:


# gridsize = (3, 2) # 3 rows, 2 cols
# fig = plt.figure(figsize=(12, 8))
# ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
# ax1.set_xlim(min(sp.date),max(sp.date))
# ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=1)
# fig.suptitle(ticker+ ": Price,Moving Averages & Twitter Sentimet",fontsize=14,horizontalalignment='right', verticalalignment='top')
# fig.subplots_adjust(hspace=0.4)
# price_plot_ma(ax=ax1,df=sp)
# sentiment_barplot(ax=ax2,df=tw)


# In[27]:


# gridsize = (2, 2) # 2 rows, 2 cols
# fig = plt.figure(figsize=(12, 8))
# ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1)
# ax2 = plt.subplot2grid(gridsize, (1, 0), colspan=2, rowspan=1)
# fig.suptitle(ticker+ ": Trade Volumes & Tweet Volumes",fontsize=14,horizontalalignment='right', verticalalignment='top')
# fig.subplots_adjust(hspace=0.5)
# ax1.set_xlim(min(sp.date),max(sp.date))
# ax2.set_xlim(min(tw.date),max(tw.date))
# price_plot_vol(ax=ax1, df=sp)
# sentiment_tweet_vol(ax=ax2,df=tw)


# In[28]:


# fig,ax = plt.subplots(figsize=(12, 8))
# fig.suptitle(ticker + ": Correlation Analysis "+ start+ " - " + end,fontsize=14,horizontalalignment='right', verticalalignment='top')
# ax=corr_plot(sp,tw)


# In[20]:


def price_plot_ma_v2(ticker, start, end, sp):
    # sp = subset_prices(d, ticker, start, end)
    fig, ax = plt.subplots(figsize=(12,8))
    fig.suptitle(ticker+ ": Price,Moving Averages",fontsize=14,horizontalalignment='right', verticalalignment='top')
    price_plot_ma(ax=ax,df=sp)
    fig.savefig('static/'+ticker+"_"+start+"_"+end+"_"+"price_ma"+".png")


# In[30]:


# price_plot_ma_v2('AAPL', '2015-01-01', '2020-12-31')


# In[21]:


# def get_sentiment_v2(tweets,ticker='TSLA',start='2017-01-01',end='2017-02-01'):
#     #subset
#     df=tweets.loc[((tweets.ticker_symbol==ticker)&(tweets.date>=start)&(tweets.date<=end))]
    
#     # apply TextBlob to body column
#     df['score'] = df['body'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
#     # create label
#     df['label'] = pd.cut(df['score'], bins=[-1, -0.66, 0.32, 1], labels=["bad", "neutral", "good"])
    
#     df=df.loc[:,["date","score","label","tweet_id","body"]]
#     return df


# In[22]:


def get_sentiment_all(tweets):
    #sbuset
    df=tweets
    # apply the SentimentIntensityAnalyzer
    df.loc[:,('score')]=df.loc[:,'body'].apply(lambda x: sia.polarity_scores(x)['compound'])
    # create label
    #bins= pd.interval_range(start=-1, freq=3, end=1)
    df.loc[:,('label')]=pd.cut(np.array(df.loc[:,'score']),bins=[-1, -0.66, 0.32, 1],right=True ,labels=["bad", "neutral", "good"])
    
    df=df.loc[:,["date","ticker_symbol","score","label","tweet_id","body"]]
    return df


# In[32]:


# sp=subset_prices(d,ticker,start,end) #get price info
tw_all = get_sentiment_all(tweets) # get tweets


# In[23]:


def price_ma_sentiment_plot(ticker, start, end, sp, tw_all=tw_all):
    # sp=subset_prices(d,ticker,start,end) #get price info
    # tw=get_sentiment_v2(tweets,ticker,start,end) # get tweets
    tw = tw_all[tw_all['ticker_symbol']==ticker]
    gridsize = (3, 2) # 3 rows, 2 cols
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
    ax1.set_xlim(min(sp.date),max(sp.date))
    ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=1)
    fig.suptitle(ticker+ ": Price,Moving Averages & Twitter Sentimet",fontsize=14,horizontalalignment='right', verticalalignment='top')
    fig.subplots_adjust(hspace=0.4)
    price_plot_ma(ax=ax1,df=sp)
    sentiment_barplot(ax=ax2,df=tw)
    fig.savefig('static/'+ticker+"_"+start+"_"+end+"_"+"price_ma_sentiment"+".png")


# In[34]:


# price_ma_sentiment_plot('AAPL', '2015-01-01', '2020-12-31', sp, tw)


# In[24]:


def trade_tweet_vol_plot(ticker, start, end, sp, tw_all=tw_all):
    tw = tw_all[tw_all['ticker_symbol']==ticker]
    gridsize = (2, 2) # 2 rows, 2 cols
    fig = plt.figure(figsize=(12, 8))
    ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=1)
    ax2 = plt.subplot2grid(gridsize, (1, 0), colspan=2, rowspan=1)
    fig.suptitle(ticker+ ": Trade Volumes & Tweet Volumes",fontsize=14,horizontalalignment='right', verticalalignment='top')
    fig.subplots_adjust(hspace=0.5)
    ax1.set_xlim(min(sp.date),max(sp.date))
    ax2.set_xlim(min(tw.date),max(tw.date))
    price_plot_vol(ax=ax1, df=sp)
    sentiment_tweet_vol(ax=ax2,df=tw)
    fig.savefig('static/'+ticker+"_"+start+"_"+end+"_"+"trade_tweet_vol"+".png")


# In[36]:


# trade_tweet_vol_plot('AAPL', '2015-01-01', '2020-12-31', sp, tw)


# In[26]:


def gen_corr_plot(ticker, start, end, sp, tw_all):
    tw = tw_all[tw_all['ticker_symbol']==ticker]
    fig,ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(ticker + ": Correlation Analysis "+ start+ " - " + end,fontsize=14,horizontalalignment='right', verticalalignment='top')
    ax=corr_plot(sp,tw)
    fig.savefig('static/'+ticker+"_"+start+"_"+end+"_"+"corr_plot"+".png")


# In[38]:


# gen_corr_plot('AAPL', '2015-01-01', '2020-12-31')


# In[25]:


# def get_sentiment_v3(tweets, ticker='TSLA', start='2017-01-01', end='2017-02-01'):
#     # Subset
#     df = tweets.loc[((tweets.ticker_symbol == ticker) & (tweets.date >= start) & (tweets.date <= end))]
    
#     # Convert to dask dataframe
#     df = dd.from_pandas(df, npartitions=4)
    
#     # Apply TextBlob to body column
#     df['score'] = df['body'].apply(lambda x: TextBlob(x).sentiment.polarity, meta=('score', 'f8'))
    
#     # Compute sentiment scores
#     df = df.compute()
    
#     # Create label
#     df['label'] = pd.cut(df['score'], bins=[-1, -0.66, 0.32, 1], labels=["bad", "neutral", "good"])
    
#     df = df.loc[:,["date","score","label","tweet_id","body"]]
#     return df


# In[40]:


# tw = get_sentiment_v3(tweets,'AAPL', '2015-01-01', '2020-12-31')


# In[41]:


# tw1 = get_sentiment_v2(tweets,'AAPL', '2015-01-01', '2020-12-31')


# In[42]:


# tw2 = get_sentiment(tweets,'AAPL', '2015-01-01', '2020-12-31')


# In[29]:


# sp = subset_prices(d, 'AAPL', '2015-01-01', '2020-12-31')


# In[30]:


# sp = subset_prices(d, 'AAPL', '2015-01-01', '2020-12-31')
# trade_tweet_vol_plot('AAPL', '2015-01-01', '2020-12-31', sp, tw)


# In[31]:


# trade_tweet_vol_plot('AAPL', '2015-01-01', '2020-12-31', sp, tw1)


# In[32]:


# trade_tweet_vol_plot('AAPL', '2015-01-01', '2020-12-31', sp, tw2)


# In[33]:


# tw_all = get_sentiment_all(tweets)


# In[34]:


# tw_all


# In[35]:


# tw_all[tw_all['ticker_symbol']=='TSLA']


# In[36]:


# def price_ma_sentiment_plot_v2(ticker, start, end, sp, tw_all):
#     # sp=subset_prices(d,ticker,start,end) #get price info
#     # tw=get_sentiment_v2(tweets,ticker,start,end) # get tweets
#     tw = tw_all[tw_all['ticker_symbol']==ticker]
#     gridsize = (3, 2) # 3 rows, 2 cols
#     fig = plt.figure(figsize=(12, 8))
#     ax1 = plt.subplot2grid(gridsize, (0, 0), colspan=2, rowspan=2)
#     ax1.set_xlim(min(sp.date),max(sp.date))
#     ax2 = plt.subplot2grid(gridsize, (2, 0), colspan=2, rowspan=1)
#     fig.suptitle(ticker+ ": Price,Moving Averages & Twitter Sentimet",fontsize=14,horizontalalignment='right', verticalalignment='top')
#     fig.subplots_adjust(hspace=0.4)
#     price_plot_ma(ax=ax1,df=sp)
#     sentiment_barplot(ax=ax2,df=tw)
#     fig.savefig('static/'+ticker+"_"+start+"_"+end+"_"+"price_ma_sentiment"+".png")


# In[37]:


# price_ma_sentiment_plot('TSLA', '2015-01-01', '2020-12-31', sp, tw_all)

