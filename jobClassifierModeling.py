import spacy
from gensim.parsing.preprocessing import strip_tags
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

my_stop_words = text.ENGLISH_STOP_WORDS.union(["nan"])
params = {'figure.figsize': (15, 10),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large'}
pylab.rcParams.update(params)
import warnings
warnings.filterwarnings("ignore")
import tqdm

"""
class SpacyTokenizer():
    '''
    Custom tokenizer
    '''
    def __init__(self, model='en_core_web_sm'):
        self.model = model
        self.nlp = spacy.load(model)
        self.spacy_tokenizer = self.nlp.tokenizer(self.nlp)
        self.stop_words = self.nlp.Defaults.stop_words
        
    def tokenizer(self, input_text: str):
        '''
        Preprocess and split text into tokens.
        
        Parameters:
        
        input_text: str
            Input text.
            
        Returns:
            np.ndarray: list of tokens
        '''
        text = strip_tags(input_text.lower())
        text = re.sub(r"[^A-Za-z]", " ", text)

        spacy_tokens = self.spacy_tokenizer(text)
        tokens = [token.lemma_.strip() for token in spacy_tokens 
                  if (len(token.lemma_.strip()) > 1) and not token.lemma_.strip() in self.stop_words]

        return tokens  
"""
def get_top_n_words(corpus: np.ndarray, n: int=5, ngram_range: tuple=(1,3)):

    tf_idf_vec = TfidfVectorizer(ngram_range=ngram_range,stop_words=my_stop_words)
    tf_idf_vec.fit(corpus.values.astype('U'))

    bag_of_words = tf_idf_vec.transform(corpus.values.astype('U'))

    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  tf_idf_vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    
    return words_freq[:n]

def tune_pipeline(pipeline: sklearn.pipeline, parameters: np.ndarray, X: pd.Series, y: pd.Series, n_splits: int=2):

    grid_search = GridSearchCV(pipeline, parameters, 
        cv=StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42),
        n_jobs=-1, verbose=10)
    
    grid_search.fit(X.values.astype('U'), y.values.astype('U'))
    
    return grid_search.best_estimator_

dataset_df = pd.read_csv('/Users/aadhi/Downloads/SMLPROJECT/Datasets/dataJobs.csv')
jobs_df = dataset_df
print(f'Dataset size: {dataset_df.shape}')
#dataset_df = dataset_df.drop_duplicates('job-description',keep='last')
#print(f'Dataset size: {dataset_df.shape}')
dataset_df['job-title'].value_counts().plot(kind='bar')
plt.title('Job descriptions per job category')
plt.show()

word_freq_dict = {}
for query in tqdm.tqdm(dataset_df['job-title'].unique()):
    word_freq_dict[query] = get_top_n_words(dataset_df[dataset_df['job-title']==query]['job-description'], n=10, ngram_range=(1,4))


for query in word_freq_dict:
    stat_string = "\n".join([f"{word_freq[0]:35} {word_freq[1]:.2f}" for word_freq in word_freq_dict[query]])
    print(f'''
===
{query}

{stat_string}
    ''')

def split_train_test(dataset_df, y, test_size=0.2):
    # remove types occur only ones
    temp_df = dataset_df[y.isin(y.value_counts()[y.value_counts()>1].index)]
    return train_test_split(temp_df, stratify=y, test_size=test_size, random_state=42)

train, test = split_train_test(dataset_df, dataset_df['job-title'], test_size=0.2)
split_df = pd.concat([train['job-title'].value_counts(), test['job-title'].value_counts()], axis=1)
split_df.columns = ['train', 'test']

split_df.plot(kind='bar', stacked=True)
plt.title('Job descriptions per job category')
plt.show()

print(f"train/test: {split_df['train'].sum()}/{split_df['test'].sum()}")
print(split_df)
#NB
print('Naive Bayes')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words, max_features=2000)),
    ('clf', MultinomialNB(fit_prior=True, class_prior=None)),
])

parameters = {
    'tfidf__ngram_range': [(1, 2)],
    'clf__alpha': (1e-2, 1e-3)
}

nb_pipeline = tune_pipeline(pipeline, parameters, train['job-description'], train['job-title'], n_splits=5)
print(nb_pipeline.steps)
#joblib.dump(nb_pipeline, './models/nb_pipeline.joblib')

pred = nb_pipeline.predict(test['job-description'].values.astype('U'))
print(classification_report(test['job-title'], pred))


#LOGISTTIC
print('Logistic')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words, max_features=2000)),
    ('clf', LogisticRegression(solver='sag')),
])
parameters = {
    'tfidf__ngram_range': [(1, 2)],
    "clf__C": [0.01, 0.1, 1],
    "clf__class_weight": ['balanced', None],
}

logistic_regression_pipeline = tune_pipeline(pipeline, parameters, train['job-description'], train['job-title'], n_splits=5)
print(logistic_regression_pipeline.steps)
#joblib.dump(logistic_regression_pipeline, './models/logistic_regression_pipeline.joblib')

pred = logistic_regression_pipeline.predict(test['job-description'].values.astype('U'))
print(classification_report(test['job-title'], pred))

#DecisionTree
print('Decisiontree')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words, max_features=2000)),
    ('clf', DecisionTreeClassifier()),
])
parameters = {
    'tfidf__ngram_range': [(1, 2), (1, 3)],
    "clf__class_weight": ['balanced', None],
}

decision_tree_pipeline = tune_pipeline(pipeline, parameters, train['job-description'], train['job-title'], n_splits=5)
print(decision_tree_pipeline.steps)
#joblib.dump(decision_tree_pipeline, './models/decision_tree_pipeline.joblib')

pred = decision_tree_pipeline.predict(test['job-description'].values.astype('U'))
print(classification_report(test['job-title'], pred))


#Linear SVC
print('linearSVC')
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=my_stop_words, max_features=2000)),
    ('clf', LinearSVC(multi_class='ovr'))
])

parameters = {
    'tfidf__ngram_range': [(1, 2)],
    "clf__C": [0.01, 0.1, 1],
    "clf__class_weight": ['balanced', None],
}

linear_svc_pipeline = tune_pipeline(pipeline, parameters, train['job-description'], train['job-title'], n_splits=5)
print(linear_svc_pipeline.steps)
#joblib.dump(linear_svc_pipeline, './models/linear_svc_pipeline.joblib')

pred = linear_svc_pipeline.predict(test['job-description'].values.astype('U'))
print(classification_report(test['job-title'], pred))


#Specific Job recommendation
jobs_df = jobs_df.drop_duplicates('job-description',keep='last')
user_qualifications = "PROFESSIONAL EXPERIENCE Universiti Teknologi PETRONAS, Perak, Malaysia June 2021 - March 2022 Research Intern (Data Science) - Python, Pandas, NumPy, Scikit-learn, TensorFlow, TF-Lite, Keras, RPi. • Designed a novel Artificial Neural Network (ANN) architecture to fabricate a Motor Bearing Non-Invasive Fault Testing Rig (MOBIT) which improves plant efficiency by 25% and decreases replacement costs by over 35%.• Fabricated a computationally light image classification algorithm to analyze Park Vector images of IM motors.• Researched with real-world induction motor data provided by Petronas, to implement predictive maintenance in industrial plants. Achieved 98.7% accuracy in fault prediction and deployed predictive model in RPi. Biosthra, Chennai, India March 2021 - December 2021 IoT Engineer and Data Analyst - Flutter, Python, Pandas, NumPy TensorFlow, Keras, Time Series, DoE.• Architected the IoT system of an automated compost pit to monitor key parameters such as pH values and temperature.• Enabled visualization of all key metrics in a live mobile dashboard using ThingSpeak. Further, built the framework tocontrol working of the compost pit via a custom flutter application, thus reducing compost time by 33%.• Developed a machine learning algorithm to predict heat produced by the compost pit at different time stamps, compostconsistencies and reduce its reliance on sensors which are bound to fail in hostile conditions.Saint Louis University, St. Louis, Missouri (Virtual) October 2021 - November 2021 Data Analyst intern - Tableau, R (Rshiny), MS Office suite.• Analyzed real-world data from client industry Ad-campaigns to decide upon the best performers.• Took charge as intern project lead and orchestrated working of the team in visualization of important metrics.• Undertook responsibility to present conclusions and visualizations from the trained model."
vectorizer = TfidfVectorizer(stop_words=my_stop_words, max_features=2000)
job_desc_matrix = vectorizer.fit_transform(jobs_df['job-description'].values.astype('U'))
user_qual_vector = vectorizer.transform([user_qualifications])
similarity_scores = cosine_similarity(job_desc_matrix, user_qual_vector)
jobs_df['similarity_score'] = similarity_scores
top_jobs = jobs_df.sort_values(by='similarity_score', ascending=False).head(15)
for i, row in top_jobs.iterrows():
    print(f"Job Title: {row['job-title']}\nJob Description: {row['job-description']}\nSimilarity Score: {row['similarity_score']}\n")
