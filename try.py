from twitterscraper import query_tweets
import datetime as dt
import pandas as pd

# paramters to define a tweet
begin_date = dt.date(2010, 5, 1)
end_date = dt.date(2020, 5, 1)
lang = 'english'
limit = 100000

# creating a query
tweets = query_tweets("amazon locker",
                      begindate=begin_date,
                      enddate=end_date,
                      limit=limit, lang=lang)

# Creating a dataframe
data = pd.DataFrame(t.__dict__ for t in tweets)