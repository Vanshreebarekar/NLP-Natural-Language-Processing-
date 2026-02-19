#!/usr/bin/env python
# coding: utf-8

# In[1]:


# pip install beautifulsoup4  # web scraping, extract data from HTML and XML files
# pip install lxml #actual work of reading the code.


# In[2]:


import bs4 as bs
import urllib  #handling URLs and making network requests


# In[3]:


import urllib.request

url="https://en.wikipedia.org/wiki/Sachin_Tendulkar"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"}
req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        content = response.read()
        print(content)
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")


# In[4]:


soup=bs.BeautifulSoup(content,'lxml')


# In[5]:


soup


# In[6]:


text=""
for paragraph in soup.find_all('p'):
    text+=paragraph.text


# In[7]:


text # standard format somehow


# In[8]:


import re


# In[9]:


import re
text = re.sub(r'\[[0-9]*\]', ' ',text) # replace number


# In[10]:


text


# In[11]:


text = re.sub(r'\s+',' ',text)


# In[12]:


text


# In[13]:


clean_text = text.lower()
clean_text = re.sub(r'\W',' ',clean_text)
clean_text = re.sub(r'\d',' ',clean_text)
clean_text = re.sub(r'\s+[a-z]\s+',' ',clean_text)
clean_text = re.sub(r'\s+',' ',clean_text)


# In[14]:


clean_text


# In[15]:


import nltk


# In[16]:


sentences = nltk.sent_tokenize(text)


# In[17]:


sentences


# In[18]:


len(sentences)


# # stop word

# In[19]:


stop_words = nltk.corpus.stopwords.words('english')
stop_words


# In[20]:


word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1


# In[21]:


word2count


# In[22]:


word2count.values()


# In[23]:


max(word2count.values())


# In[24]:


for key in word2count.keys():
    word2count[key] = word2count[key]/(max(word2count.values()))


# In[25]:


word2count


# In[26]:


sent2score = {}
for sentence in sentences: # 'He inspired movements for civil rights and freedom across the world.',
    for word in nltk.word_tokenize(sentence.lower()): # [He ,inspired, movements ,for civil rights and freedom across the world.,]
        if word in word2count.keys(): # He
            if len(sentence.split(' '))<50:
                if sentence not in sent2score.keys():
                    sent2score[sentence] = word2count[word]
                else:
                    sent2score[sentence] += word2count[word]


# In[27]:


sent2score


# In[28]:


import heapq # priority queue algorithm


# In[29]:


best_sentences = heapq.nlargest(20,sent2score,key=sent2score.get)
best_sentences


# In[30]:


suumary = ' '.join(best_sentences)


# In[31]:


suumary


# In[ ]:




