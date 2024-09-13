# MeToo Tweets Sentiment Analysis and Topic Modeling
Overview
This project involves analyzing tweets related to the #MeToo movement. The analysis includes data cleaning, sentiment analysis using NLTK's VADER, word cloud visualization, tokenization, vectorization, and topic modeling using Latent Dirichlet Allocation (LDA). The project also identifies the most influential users based on retweets and mentions.

# Installation
To run this project, you need to have Python installed along with the following libraries:


pip install nltk pandas matplotlib wordcloud scikit-learn textblob
Additional NLTK Downloads
Some NLTK resources need to be downloaded before running the code:

python
Copy code
import nltk
nltk.download('punkt')
nltk.download('stopwords')
Data Preprocessing
# 1. Loading the Data
The dataset consists of tweets related to the #MeToo movement, stored in a CSV file named MeToo_tweets.csv. The data is loaded into a pandas DataFrame:

python
Copy code
df = pd.read_csv('MeToo_tweets.csv')
2. Cleaning the Data
A custom function cleantext is defined to clean the tweet text by removing mentions, hashtags, URLs, punctuation, emojis, and stopwords:


df['Text_clean'] = df['Text'].apply(cleantext)
3. Removing Duplicates
To ensure data quality, duplicate tweets are removed:


df = df.drop_duplicates()
4. Tokenization
The cleaned text is tokenized using NLTK's word_tokenize:


df['Tokenized_Text'] = df['Text_clean'].apply(lambda text: word_tokenize(text.lower()))
Sentiment Analysis
Sentiment analysis is performed using the SentimentIntensityAnalyzer from NLTK's VADER:

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
df['Sentiment'] = df['Text_clean'].apply(lambda text: sid.polarity_scores(text)['compound'])
Word Cloud Visualization
A word cloud is generated to visualize the most common words in the cleaned tweet text:

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df['Text_clean']))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
Topic Modeling
1. Vectorization
The tokenized text is converted into a document-term matrix using CountVectorizer:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['Tokenized_Text'].apply(lambda tokens: ' '.join(tokens)).tolist())
2. Latent Dirichlet Allocation (LDA)
LDA is used to identify topics within the tweets:


from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=20, random_state=42)
lda.fit(X)
3. Word Cloud for Topics
A combined word cloud is generated to visualize the top words from all topics:

python
Copy code
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(all_keywords))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
Influential User Identification
The project also attempts to identify the most influential user based on the number of retweets and mentions:

python
Copy code
# Calculate influence and identify the most mentioned screen name
db['influence'] = db['Retweet_count'] + db['Favorite_count'].notnull().astype(int)
most_mentioned_screenname = max(set(all_mentioned_screennames), key=all_mentioned_screennames.count)
most_influential_user = db[db['screenname_mentioned'].apply(lambda x: most_mentioned_screenname in x)].iloc[0]

print("Most Influential User:")
print("Mentioned Screen Name:", most_mentioned_screenname)
print("Retweets:", most_influential_user['Retweet_count'])
Note: Make sure the DataFrame db is defined correctly before running the code. Replace db with the correct DataFrame name if necessary.

# Results
Sentiment Analysis: Provided a sentiment score for each tweet.
Word Cloud: Visualized the most common words in the tweets.
Topic Modeling: Identified 20 topics from the tweet corpus.
Influential User Identification: Identified the most influential user based on retweets and mentions.
License
This project is licensed under the MIT License.
