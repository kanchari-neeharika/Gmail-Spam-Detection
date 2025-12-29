import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
df = pd.read_csv("C:\\Users\\nehak\\Downloads\\mail_data.csv", encoding='latin-1')
spam_df = df[df['Category'] == 'spam']
spam_messages = spam_df['Message']
count_vectorizer = CountVectorizer(stop_words='english',lowercase=True,min_df=1)
spam_features = count_vectorizer.fit_transform(spam_messages)
feature_names = count_vectorizer.get_feature_names_out()
word_counts = spam_features.sum(axis=0)
word_counts_list = word_counts.tolist()[0]
word_freq_df = pd.DataFrame({'word': feature_names,'count': word_counts_list})
top_words = word_freq_df.sort_values(by='count', ascending = False).head(20)
plt.figure(figsize=(12, 8))
plt.barh(top_words['word'], top_words['count'], color = 'indianred')
plt.gca().invert_yaxis()
plt.title('Top 20 Most Frequent Words in Spam Emails (Excluding Stop Words)', fontsize=14)
plt.xlabel('Frequency Count', fontsize=12)
plt.ylabel('Word', fontsize=12)
for index, value in enumerate(top_words['count']):
    plt.text(value, index, str(value), ha = 'left', va = 'center', fontsize=9)
plt.grid(axis='x', linestyle='--', alpha = 0.6)
plt.tight_layout()
plt.show()