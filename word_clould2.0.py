#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from wordcloud import STOPWORDS as WORDCLOUD_STOPWORDS
from wordcloud import WordCloud
import jieba
import streamlit as st
import contextlib, io

resources = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
    "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
}

for package, lookup_path in resources.items():
    try:
        nltk.data.find(lookup_path)
    except LookupError:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            nltk.download(package, quiet=True, raise_on_error=False)

csv_path = 'food.csv'
df = pd.read_csv(csv_path)

custom_terms = ['水潤餅', '柴梳餅', '竹塹餅', '銀耳蓮子湯', '麻糬紅豆Q餅', '水蒸蛋糕']
for term in custom_terms:
    jieba.add_word(term)
with open("chinese_stopwords.txt", "r", encoding="utf-8") as f:
    STOPWORDS = f.read().splitlines()

df['clean_text'] = df['介紹'].fillna('').astype(str)

df['clean_text'] = df['clean_text'].str.replace(r'\\n', ' ', regex=True)

df['clean_text'] = df['clean_text'].str.replace(
    r'[^一-龥A-Za-z\s]',
    ' ',
    regex=True
)

raw_tokens = [jieba.lcut(x) for x in df['clean_text']]
df['raw_tokens'] = raw_tokens

df['clean_tokens'] = [
    [token.strip().lower() for token in token_list]
    for token_list in df['raw_tokens']
]

df['tokens'] = [
    [
        token
        for token in token_list
        if token != '' and token not in STOPWORDS
    ]
    for token_list in df['clean_tokens']
]

all_tokens = []
for tokens in df['tokens']:
    for token in tokens:
        all_tokens.append(token)
        
top_terms = pd.DataFrame(Counter(all_tokens).most_common(20), columns=['term', 'count'])

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_cjk_font():
    candidates = [
        'Noto Sans CJK TC',
        'Noto Sans CJK SC',
        'Noto Sans CJK JP',
        'Microsoft JhengHei',
        'PingFang TC',
        'SimHei',
        'Arial Unicode MS'
    ]

    available_fonts = fm.fontManager.ttflist

    for font in candidates:
        for f in available_fonts:
            if font == f.name:
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False

                print(f'Using font: {font}')
                print(f'Font path: {f.fname}')

                return f.fname

    print('No CJK font found by matplotlib.')
    return None

font_path = setup_cjk_font()

st.title("新竹水潤餅文字雲")

num_words = st.slider(
    "選擇文字數量",
    min_value=1,
    max_value=1000,
    value=100,
    step=1
)

filtered_terms = top_terms.head(num_words)

freq_dict = dict(zip(
    filtered_terms['term'],
    filtered_terms['count']
))

wc = WordCloud(
    font_path=font_path,
    width=1000,
    height=600,
    background_color='white',
    collocations=False
).generate_from_frequencies(freq_dict)

fig, ax = plt.subplots(figsize=(12, 7))

ax.imshow(wc)
ax.axis('off')
st.pyplot(fig)


# In[ ]:




