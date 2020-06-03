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
from nltk.stem import PorterStemmer

stemming = PorterStemmer()
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer

# nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

lemmatizer = nltk.stem.WordNetLemmatizer()
wordnet_lemmatizer = WordNetLemmatizer()

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

data.drop(['username', 'user_id', 'tweet_id', 'timestamp', 'retweets', 'screen_name', 'text_html', 'tweet_url', 'links',
           'hashtags', 'timestamp_epochs',
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
count_total = 0
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
        count_total = count_total + 1
        data.loc[i, 'Address'] = 1


    elif (sent.sentiment.polarity < 0):
        count_neg = count_neg + 1
        count_total = count_total + 1
        data.loc[i, 'Address'] = -1


    else:
        count_neut += 1
        count_total = count_total + 1
        data.loc[i, 'Address'] = 0

print("Total tweets with sentiment:", count_total)
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

#data['Text'] = data.apply(lambda row: nltk.word_tokenize(row['Lemmatize']), axis=1)

'''
timing = ['ontime','beforetime','before','within','withintime']
space = ['space','storage','large','huge','spacious','big']
functionality = ['comfort','good','handy','pickup','easy function','convinient','smooth','easy',
                 'great','recommend','love','satisfy','happy','awesome']
security = ['secure','lock','trust','safe','locksystem','codeword']
location = ['early','nearby','close','marketplace','walking','less']
charge = ['money','charge','affordable','price','minimal','minimum','less']

final_res = {"timing1" : 0,'space1' : 0,'functionality1' : 0,'security1': 0,'location1': 0,'charge1': 0}






for r1 in res1:
    if r1 in timing:
        final_res['timing1'] += 1
    elif r1 in space:
        final_res['space1'] += 1
    elif r1 in functionality:
        final_res['functionality1'] += 1
    elif r1 in security:
        final_res['security1'] += 1
    elif r1 in location:
        final_res['location1'] += 1
    elif r1 in charge:
        final_res['charge1'] += 1
    else:
        continue

print(final_res)

'''

'''


words = nltk.tokenize.word_tokenize(a)

#w1 = [stemming.stem(word) for word in words]

remove1 = ['amazon','delivery','not','I','the']   
tokens = [token for token in words if token.isalpha() and token not in remove1]  # filter alphabets

fdist = nltk.FreqDist(tokens)





t1 = {}
for t in timing:
    t1[t] = fdist[t]
print(t1)


s1 = {}
for s in space:
    s1[s] = fdist[s]
print(s1)


f1 = {}
for f in functionality:
    f1[f] = fdist[f]
print(f1)


s1 = {}
for s in security:
    s1[s] = fdist[s]
print(s1)


l1 = {}
for l in location:
    l1[l] = fdist[l]
print(l1)


c1 = {}
for c in charge:
    c1[c]= fdist[c]
print(c1)



'''

"""

rslt = pd.DataFrame(fdist.most_common(top_N),
                    columns=['Word', 'Frequency'])

#rslt.to_csv('20_most_common.csv')


sample = ['good','bad','worst','best','happy','sad','delivery','location','storage','shipping','package',
          'service','space','timing']
freq1 = {}

for t1 in sample:
        freq1[t1] = fdist[t1]

dfObj = pd.DataFrame(list(freq1.items()))

#dfObj.to_csv('Freq_of_keywords.csv')     

"""

