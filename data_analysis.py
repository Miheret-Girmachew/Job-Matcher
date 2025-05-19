import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import re
import string
import nltk
import joblib
import os

warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

import seaborn as sns
from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def setup_nltk_local_path_priority():
    local_nltk_path = os.path.join(os.getcwd(), 'nltk_data_local')
    if not os.path.exists(local_nltk_path):
        print(f"WARNING: Local NLTK path {local_nltk_path} does not exist. Lemmatization might fail.")
        print("Ensure NLTK resources are correctly placed in 'nltk_data_local'.")
        return False
    if local_nltk_path not in nltk.data.path:
        nltk.data.path.insert(0, local_nltk_path)
        print(f"Prioritized local NLTK data path: {local_nltk_path}")
    else:
        print(f"Local NLTK data path already in search paths: {local_nltk_path}")
    try:
        nltk.data.find('corpora/wordnet')
        nltk.data.find('corpora/omw-1.4')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        print("Essential NLTK resources found in search paths.")
        return True
    except LookupError as e:
        print(f"WARNING: Still cannot find an NLTK resource: {e}")
        return False

print("Setting up NLTK resource paths...")
if not setup_nltk_local_path_priority():
    print("NLTK setup might be incomplete.")
print("NLTK resource path setup complete.\n")

TFIDF_MAX_FEATURES = 750
TFIDF_MIN_DF = 3
SELECTED_MODEL = 'LogisticRegression'

print("Loading dataset...")
try:
    resumeDataSet = pd.read_csv('resume_dataset.csv', encoding='utf-8')
except FileNotFoundError:
    print("ERROR: 'resume_dataset.csv' not found.")
    exit()
resumeDataSet['cleaned_resume'] = ''
print("First few rows of dataset:\n", resumeDataSet.head())

print("\nDistinct categories:\n", resumeDataSet['Category'].unique())
print(f"\nTotal unique categories: {resumeDataSet['Category'].nunique()}")
category_counts = resumeDataSet['Category'].value_counts()
print("\nCategory counts:\n", category_counts)

min_class_count_for_cv = category_counts.min()
cv_folds = min(3, min_class_count_for_cv)
if cv_folds < 2:
    print(f"WARNING: Smallest class {category_counts.idxmin()} has {min_class_count_for_cv} samples. Forcing CV to 2 folds.")
    cv_folds = 2

plt.figure(figsize=(12, 10))
sns.countplot(y="Category", data=resumeDataSet, order=category_counts.index, palette="viridis")
plt.title("Resume Category Counts (Class Imbalance)", fontsize=16)
plt.xlabel("Number of Resumes", fontsize=14)
plt.ylabel("Job Category", fontsize=14)
plt.xticks(fontsize=12); plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("resume_category_counts.png"); print("Saved resume_category_counts.png")
plt.show(block=False); plt.pause(1)

plt.figure(figsize=(12, 12))
cmap = plt.get_cmap('coolwarm'); colors = [cmap(i) for i in np.linspace(0, 1, len(category_counts))]
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', shadow=True, colors=colors, startangle=90, pctdistance=0.85)
plt.axis('equal'); plt.title('CATEGORY DISTRIBUTION (Illustrating Imbalance)', fontsize=18, pad=20)
plt.savefig("resume_category_distribution_pie.png"); print("Saved resume_category_distribution_pie.png")
plt.show(block=False); plt.pause(1)

print("\nInitializing lemmatizer and stop words...")
lemmatizer = WordNetLemmatizer(); stop_words_english = set(stopwords.words('english') + ['``', "''"])

def cleanResume(resumeText):
    resumeText = re.sub(r'http\S+\s*', ' ', resumeText)
    resumeText = re.sub(r'RT|cc', ' ', resumeText)
    resumeText = re.sub(r'#\S+', '', resumeText)
    resumeText = re.sub(r'@\S+', ' ', resumeText)
    resumeText = resumeText.lower()
    resumeText = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', resumeText)
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub(r'\s+', ' ', resumeText)
    words = nltk.word_tokenize(resumeText)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english and len(word) > 2]
    return " ".join(words).strip()

print("Cleaning resumes...")
resumeDataSet['cleaned_resume'] = resumeDataSet['Resume'].apply(cleanResume)

resumeDataSet['cleaned_resume_word_count'] = resumeDataSet['cleaned_resume'].apply(lambda x: len(x.split()))
plt.figure(figsize=(12, 6))
sns.histplot(resumeDataSet['cleaned_resume_word_count'], bins=30, kde=True)
plt.title('Distribution of Resume Lengths (Word Count after Cleaning)', fontsize=16)
plt.xlabel('Word Count', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.savefig("resume_length_distribution.png"); print("Saved resume_length_distribution.png")
plt.show(block=False); plt.pause(1)

plt.figure(figsize=(12, 10))
sns.boxplot(x='cleaned_resume_word_count', y='Category', data=resumeDataSet, order=category_counts.index, palette="coolwarm")
plt.title('Resume Lengths (Word Count) by Category', fontsize=16)
plt.xlabel('Word Count', fontsize=14)
plt.ylabel('Job Category', fontsize=14)
plt.xticks(fontsize=10); plt.yticks(fontsize=10)
plt.tight_layout()
plt.savefig("resume_length_by_category_boxplot.png"); print("Saved resume_length_by_category_boxplot.png")
plt.show(block=False); plt.pause(1)

print("\nGenerating word cloud...")
all_cleaned_text = " ".join(resume for resume in resumeDataSet['cleaned_resume'])
if all_cleaned_text.strip():
    wc = WordCloud(width=1200, height=600, background_color='white', stopwords=stop_words_english, collocations=False).generate(all_cleaned_text)
    plt.figure(figsize=(15, 7.5)); plt.imshow(wc, interpolation='bilinear'); plt.axis("off")
    plt.title("Word Cloud of Cleaned Resume Texts", fontsize=16)
    plt.savefig("resume_wordcloud.png"); print("Saved resume_wordcloud.png")
    plt.show(block=False); plt.pause(1)
else:
    print("Not enough content for word cloud.")

print("\nEncoding labels..."); le = LabelEncoder()
resumeDataSet['Category_encoded'] = le.fit_transform(resumeDataSet['Category'])
joblib.dump(le, 'label_encoder.pkl'); print("Label Encoder saved. Classes:", le.classes_)

print("\nExtracting features using TF-IDF...")
requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category_encoded'].values
word_vectorizer = TfidfVectorizer(sublinear_tf=True, stop_words='english', max_features=TFIDF_MAX_FEATURES,
                                  ngram_range=(1, 2), min_df=TFIDF_MIN_DF, max_df=0.95)
word_vectorizer.fit(requiredText); WordFeatures = word_vectorizer.transform(requiredText)
joblib.dump(word_vectorizer, 'tfidf_vectorizer.pkl')
print(f"TF-IDF Vectorizer saved. Shape: {WordFeatures.shape}")

print(f"\nSplitting data... Using {cv_folds}-fold CV.")
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42,
                                                    test_size=0.25, stratify=requiredTarget)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

print(f"\n--- Training Model: {SELECTED_MODEL} ---")
stratified_kfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
if SELECTED_MODEL == 'LogisticRegression':
    model_to_tune = LogisticRegression(solver='liblinear', random_state=42, max_iter=300,
                                       multi_class='ovr', class_weight='balanced')
    param_grid = {'C': [0.1, 1, 5, 10, 50], 'penalty': ['l1', 'l2']}
else:
    print(f"Model {SELECTED_MODEL} not explicitly configured for final run. Exiting.")
    exit()

grid_search = GridSearchCV(estimator=model_to_tune, param_grid=param_grid, cv=stratified_kfold,
                           scoring='f1_macro', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_clf = grid_search.best_estimator_
print("\nBest hyperparameters:", grid_search.best_params_)
print(f"Best CV f1_macro score: {grid_search.best_score_:.4f}")
final_model_name = f'best_{SELECTED_MODEL}_model.pkl'
joblib.dump(best_clf, final_model_name); print(f"Best model saved as '{final_model_name}'.")

print("\nEvaluating model performance...")
y_train_pred = best_clf.predict(X_train)
train_f1_macro = metrics.f1_score(y_train, y_train_pred, average='macro', zero_division=0)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f'\nTraining F1 Macro: {train_f1_macro:.4f}, Training Accuracy: {train_accuracy:.4f}')

prediction = best_clf.predict(X_test)
test_f1_macro = metrics.f1_score(y_test, prediction, average='macro', zero_division=0)
test_accuracy = accuracy_score(y_test, prediction)
print(f'Test F1 Macro: {test_f1_macro:.4f}, Test Accuracy: {test_accuracy:.4f}')

print(f"\nClassification report for {SELECTED_MODEL} on test set:\n")
report_dict = classification_report(y_test, prediction, target_names=le.classes_, zero_division=0, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
print(report_df)
macro_avg_f1 = report_df.loc['macro avg', 'f1-score']
weighted_avg_f1 = report_df.loc['weighted avg', 'f1-score']
print(f"\nMacro Avg F1-Score: {macro_avg_f1:.4f}\nWeighted Avg F1-Score: {weighted_avg_f1:.4f}\nOverall Test Accuracy: {test_accuracy:.4f}")

print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, prediction, labels=np.arange(len(le.classes_)))
plt.figure(figsize=(max(12, len(le.classes_)*0.6), max(10, len(le.classes_)*0.5)))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, annot_kws={"size": 8})
plt.title(f'Confusion Matrix ({SELECTED_MODEL})', fontsize=16); plt.xlabel('Predicted Label', fontsize=14); plt.ylabel('True Label', fontsize=14)
plt.xticks(rotation=65, ha="right", fontsize=10); plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(f"confusion_matrix_{SELECTED_MODEL}.png"); print(f"Saved confusion_matrix_{SELECTED_MODEL}.png")
plt.show(block=False); plt.pause(1)

class_f1_scores = report_df[:-3]['f1-score']
class_support = report_df[:-3]['support']
fig, ax1 = plt.subplots(figsize=(14, 8))
color = 'tab:blue'
ax1.set_xlabel('Job Category', fontsize=12)
ax1.set_ylabel('F1-Score', color=color, fontsize=12)
bars = ax1.bar(class_f1_scores.index, class_f1_scores, color=color, alpha=0.7, label='F1-Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(np.arange(len(class_f1_scores.index)))
ax1.set_xticklabels(class_f1_scores.index, rotation=70, ha="right", fontsize=9)
ax1.grid(True, axis='y', linestyle='--', alpha=0.7)
for bar in bars:
    yval = bar.get_height()
    if yval > 0.001:
      plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom', fontsize=8)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Support (Test Set Count)', color=color, fontsize=12)
ax2.plot(class_support.index, class_support, color=color, marker='o', linestyle='--', label='Support (Test Count)')
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()
plt.title(f'Per-Class F1-Scores and Support ({SELECTED_MODEL} on Test Set)', fontsize=16, pad=20)
plt.savefig(f"f1_scores_vs_support_{SELECTED_MODEL}.png"); print(f"Saved f1_scores_vs_support_{SELECTED_MODEL}.png")
plt.show(block=False); plt.pause(1)

print("\nClosing all plot figures...")
plt.close('all')

print(f"\n--- Training and Analysis ({SELECTED_MODEL}, All {resumeDataSet['Category'].nunique()} classes, {TFIDF_MAX_FEATURES} features, min_df={TFIDF_MIN_DF}) Script Finished ---")