from twitterscraper import query_tweets
import datetime as dt
import pandas as pd
import csv
import nltk
# nltk.download('wordnet')
import re
import string
from nltk.corpus import stopwords
from textblob import TextBlob
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()

# paramters to define a tweet
bd = dt.date(2015, 5, 1)
ed = dt.date(2020, 5, 1)
ln = 'english'
limit = 100000

# creating a query
tweets = query_tweets("amazon locker",
                      begindate=bd,
                      enddate=ed,
                      limit=limit, lang=ln)

# Creating a dataframe
data = pd.DataFrame(t.__dict__ for t in tweets)

data.drop(['username', 'user_id', 'tweet_id', 'timestamp', 'retweets', 'screen_name', 'text_html', 'tweet_url',
           'links', 'hashtags', 'timestamp_epochs',
           'has_media', 'img_urls', 'video_url', 'likes',
           'replies', 'is_replied', 'is_reply_to', 'parent_tweet_id', 'reply_to_users'],
          axis=1, inplace=True)

# To remove url
data['text'] = data['text'].replace(to_replace=r'https?:\/\/.*[\r\n]*', value='', regex=True)

# punctuations removal
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
data['text'] = data['text'].str.replace(RE_PUNCTUATION, "")

# duplicate rows are removed
data.drop_duplicates(subset="text", keep='first', inplace=True)

data.reset_index(inplace=True, drop=True)

# For textBlob to segregate as positive, negative and neutral
count_pos = 0
count_neg = 0
count_neut = 0

address = []
for i in range(len(data)):
    address.append(0)

data['Address'] = address

for i in range(len(data)):
    sent = TextBlob(str(data.loc[i, 'text']))
    if (sent.sentiment.polarity > 0):
        count_pos = count_pos + 1
        data.loc[i, 'Address'] = 1


    elif (sent.sentiment.polarity < 0):
        count_neg = count_neg + 1
        data.loc[i, 'Address'] = -1


    else:
        count_neut += 1
        data.loc[i, 'Address'] = 0

print("positive tweets:", count_pos)
print("negative tweets:", count_neg)
print("neutral tweets:", count_neut)

# To drop negative tweets
for i in range(len(data)):
    data.drop(data[data.Address == -1].index, inplace=True)

# To remove stopwords
stop = stopwords.words('english')
pat = r'\b(?:{})\b'.format('|'.join(stop))
data['tweet'] = data['text'].str.replace(pat, '')
data['tweet'] = data['tweet'].str.replace(r'\s+', ' ')

# to drop text column
data.drop(['text'], axis=1, inplace=True)

# To remove numbers from tweet column
data['tweet'] = data['tweet'].str.replace('\d+', '')

data["tweet"] = data["tweet"].str.lower()

data.drop_duplicates(subset="tweet", keep='first', inplace=True)


# to lemmatize data
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


data['Lemmatize'] = data['tweet'].apply(lambda x: lemmatize_sentence(x))

data.drop_duplicates(subset="Lemmatize", keep='first', inplace=True)

# To tokenize column Lemmatize and form new cloumn text
data['Text'] = data.apply(lambda row: nltk.word_tokenize(row['Lemmatize']), axis=1)

data.drop(['tweet', 'Lemmatize'], axis=1, inplace=True)

# Categories to classify tweets
timing = ['ontime', 'beforetime', 'before', 'within', 'withintime', 'time', 'early']
space = ['space', 'storage', 'large', 'huge', 'spacious', 'big', 'giant']
functionality = ['comfort', 'good', 'handy', 'easy', 'convenient', 'smooth', 'pick', 'enjoy', 'clean', 'genius',
                 'excite',
                 'great', 'recommend', 'love', 'satisfy', 'happy', 'awesome', 'sharingiscaring', 'awesomeness',
                 'cool''amuse', 'surprise', 'interesting', 'use', 'locker']
security = ['secure', 'lock', 'trust', 'safe', 'locksystem', 'codeword']
location = ['nearby', 'close', 'marketplace', 'walk', 'near', 'around', 'corner',
            'locate', 'neighborhood', 'nearest', 'location']
charge = ['money', 'charge', 'affordable', 'price', 'minimal', 'minimum', 'less', 'cheap']

final_res = {"Tweets related to timing of delivery": 0,
             'Tweets related to space of lockers': 0,
             'Tweets related to functionality of lockers': 0,
             'Tweets related to security of lockers': 0,
             'Tweets related to location of lockers': 0,
             'Tweets related to charges of locker': 0}

for r1 in data['Text']:
    for i in r1:
        if i in timing:
            final_res['Tweets related to timing of delivery'] += 1
            break

        elif i in space:
            final_res['Tweets related to space of lockers'] += 1
            break

        elif i in security:
            final_res['Tweets related to security of lockers'] += 1
            break

        elif i in location:
            final_res['Tweets related to location of lockers'] += 1
            break

        elif i in charge:
            final_res['Tweets related to charges of locker'] += 1
            break

        elif i in functionality:
            final_res['Tweets related to functionality of lockers'] += 1
            break

        else:
            continue

print("Categories to classify positive tweets :")
for i, j in final_res.items():
    print(i, ':', j)



