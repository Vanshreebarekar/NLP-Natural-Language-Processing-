#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df = pd.read_csv('UpdatedResumeDataSet.csv')


# In[3]:


df


# In[ ]:





# In[4]:


def clean_resume(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', '  ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]',r' ', text) 
    text = re.sub('\s+', ' ', text)
    return text.lower().strip()


# In[5]:


df['cleaned_resume'] = df['Resume'].apply(lambda x: clean_resume(x))


# In[6]:


le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(
    df['Resume'], df['Category_Encoded'], test_size=0.2, random_state=42
)


# In[8]:


resume_pipeline= Pipeline([
    ('bow', CountVectorizer(analyzer=clean_resume)),
    ('tfidf', TfidfTransformer()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])


# In[9]:


resume_pipeline


# In[10]:


resume_pipeline.fit(X_train, y_train)


# In[11]:


tfidf = TfidfVectorizer(stop_words='english', max_features=1500)
X = tfidf.fit_transform(df['cleaned_resume'])
y = df['Category_Encoded']


# In[12]:


model = OneVsRestClassifier(KNeighborsClassifier())
model.fit(X, y)


# In[13]:


def get_job_role(input_text):
    cleaned_input = clean_resume(input_text)
    input_features = tfidf.transform([cleaned_input])
    prediction_id = model.predict(input_features)[0]
    job_role = le.inverse_transform([prediction_id])[0]
    return job_role


# In[14]:


any_resume ="""Java, Spring Boot, Microservices, Hibernate, MySQL.
Experience: 4 years of backend development."""
print(f"Predicted Role: {get_job_role(any_resume)}")


# In[15]:


any_resume="""Technical Skills: Selenium WebDriver, Java, TestNG, JIRA, Manual Testing, Bug Tracking.
Experience:
- Automated functional and regression test cases using Selenium.
- Performed Sanity and Smoke testing for every new build.
- Documented and tracked bugs in JIRA, coordinating with the development team.
Education: B.E in Electronics.
"""
print(f"Predicted Role: {get_job_role(any_resume)}")


# In[16]:


import pickle
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
print("All files (model.pkl, tfidf.pkl, label_encoder.pkl) have been saved successfully!")


# In[ ]:





# In[ ]:




