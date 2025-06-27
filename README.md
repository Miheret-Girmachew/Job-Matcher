# Intelligent Resume Screening System 📄🤖

This project presents an **Intelligent Resume Screening System** that leverages Natural Language Processing (NLP) and Machine Learning (ML) to automate the classification of resumes into predefined job categories. It features an end-to-end pipeline from data preprocessing and model training to a user-friendly web application for real-time screening.


## ✨ Features

* **Automated Resume Parsing:** Extracts text from uploaded PDF files or direct text input.
* **Advanced Text Preprocessing:** Includes cleaning (URLs, special characters), lowercasing, stop-word removal, and lemmatization using NLTK.
* **TF-IDF Feature Engineering:** Converts resume text into meaningful numerical vectors using optimized TF-IDF parameters (`max_features=750`, `min_df=3`, n-grams).
* **Machine Learning Classification:** Employs a tuned **Logistic Regression** model for robust multi-class categorization.
* **Interactive Web Application:** Built with Streamlit, providing an intuitive UI for:
    * Selecting a target job category.
    * Uploading resumes (PDF) or pasting text.
    * Viewing the model's predicted category.
    * Assessing if the prediction matches the target job category.
* **Comprehensive Evaluation:** Model performance is detailed with metrics like accuracy, F1-score (macro and weighted), precision, recall, and a confusion matrix.
* **Data Visualization:** Includes plots for category distribution, resume length analysis, word clouds, and per-class F1-scores vs. support.

## 🚀 Project Goal

To address the inefficiencies and potential biases in traditional manual resume screening by providing an automated, data-driven solution for initial candidate categorization, thereby enabling recruiters to focus on the most promising applicants more effectively.

## 🛠️ Technologies & Libraries

* **Python 3.x**
* **Core ML & Data Science:**
    * `scikit-learn`: For TF-IDF, Logistic Regression, model evaluation, train-test split, GridSearchCV.
    * `pandas`: For data manipulation.
    * `numpy`: For numerical operations.
* **NLP:**
    * `nltk`: For tokenization, stop-word removal, lemmatization (WordNet).
* **PDF Processing:**
    * `pdfplumber`: Primary PDF text extraction.
    * `PyPDF2`: Fallback PDF text extraction.
* **Web Application:**
    * `streamlit`: For building the interactive UI.
* **Visualization:**
    * `matplotlib`: For base plotting.
    * `seaborn`: For enhanced statistical plots.
    * `wordcloud`: For generating word clouds.
* **Model Persistence:**
    * `joblib`: For saving and loading trained models and preprocessors.
* **Environment:**
    * `venv`: For managing Python virtual environments.

## Dataset

The system was trained and evaluated on `resume_dataset.csv`, a publicly available dataset containing 169 resumes across 25 distinct job categories. A key characteristic of this dataset is its significant class imbalance, which influenced modeling choices and evaluation strategies (e.g., using `class_weight='balanced'` and `f1_macro` scoring).

## ⚙️ Methodology Overview

1. **Data Loading & EDA:** Initial analysis to understand resume content and category distribution. Visualized class imbalance.
2. **Text Preprocessing:** A custom pipeline (`cleanResume`) was developed for:
    * Noise removal (URLs, RTs, mentions, hashtags, punctuation, non-ASCII).
    * Lowercase conversion.
    * Tokenization.
    * Stop-word filtering.
    * Lemmatization.
3. **Feature Engineering:** `TfidfVectorizer` was used with `max_features=750`, `min_df=3`, `ngram_range=(1,2)`, and `sublinear_tf=True`.
4. **Model Selection & Training:**
    * Compared KNN, Naive Bayes, and Logistic Regression.
    * **Logistic Regression** with `class_weight='balanced'` was selected as the final model.
    * Hyperparameters tuned using `GridSearchCV` with `StratifiedKFold` (cv=3) and `f1_macro` scoring.
    * Optimal parameters: `C=10`, `penalty='l1'`.
5. **Evaluation:** Assessed using accuracy, precision, recall, F1-score (per-class, macro, weighted), and confusion matrix.
6. **Application Development:** A Streamlit app (`app.py`) was built to deploy the trained model for interactive screening.


## 🔧 Setup and Installation

1. **Clone the repository (if applicable):**
    ```bash
    git clone https://github.com/Miheret-Girmachew/Job-Matcher
    cd Job-Matcher
    ```

2. **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **NLTK Resource Setup:**
    The project utilizes a local `nltk_data_local` directory for NLTK resources (stopwords, punkt, wordnet, omw-1.4).
    * When you first run `python data_analysis.py`, it will attempt to download these resources into the `nltk_data_local` folder.
    * **Crucial Manual Step (If downloads are incomplete):** If `wordnet.zip` or `omw-1.4.zip` are present in `nltk_data_local/corpora/` but not unzipped, you **must manually unzip them** directly into `nltk_data_local/corpora/wordnet/` and `nltk_data_local/corpora/omw-1.4/` respectively. Ensure there are no extra nested folders (e.g., `nltk_data_local/corpora/wordnet/wordnet/`). The script `data_analysis.py` will confirm if essential resources are found.

---

## 🚀 Running the Project

### 1. Model Training and EDA (`data_analysis.py`)
This script performs data loading, exploratory data analysis, text preprocessing, feature engineering, model training (Logistic Regression by default), evaluation, and saves the trained model artifacts and various plots.

```bash
python data_analysis.py
```

#### **Output Artifacts:**
- `best_LogisticRegression_model.pkl`: The trained classification model.
- `tfidf_vectorizer.pkl`: The fitted TF-IDF vectorizer.
- `label_encoder.pkl`: The fitted label encoder.
- Various `.png` image files for EDA and evaluation plots (e.g., `confusion_matrix_LogisticRegression.png`, `f1_scores_vs_support_LogisticRegression.png`).

---

### 2. Streamlit Web Application (`app.py`)
This script launches the interactive resume screening application. **Ensure that `data_analysis.py` has been run successfully at least once to generate the required `.pkl` model artifacts.**

```bash
streamlit run app.py
```

Open the local URL (usually [http://localhost:8501](http://localhost:8501)) displayed in your terminal in a web browser.

---

## 📁 File Structure

```
Resume-Screening/
├── app.py                           # Streamlit application
├── data_analysis.py                 # Model training and EDA script
├── resume_dataset.csv               # Dataset
├── requirements.txt                 # Python dependencies
├── nltk_data_local/                 # Local NLTK data (auto-populated or manually set up)
│   ├── corpora/
│   │   ├── wordnet/
│   │   ├── omw-1.4/
│   │   └── ...
│   └── tokenizers/
│       └── punkt/
├── best_LogisticRegression_model.pkl  # Saved model artifact
├── tfidf_vectorizer.pkl             # Saved vectorizer
├── label_encoder.pkl                # Saved label encoder
├── resume_category_counts.png       # Saved plot
├── resume_category_distribution_pie.png # Saved plot
├── resume_length_distribution.png   # Saved plot
├── resume_length_by_category_boxplot.png # Saved plot
├── resume_wordcloud.png             # Saved plot
├── confusion_matrix_LogisticRegression.png # Saved plot
├── f1_scores_vs_support_LogisticRegression.png # Saved plot
└── README.md                        # This file
```

---

## 🎯 Key Challenges & Learnings

- **Data Scarcity & Imbalance:** The primary challenge was the small dataset size (169 samples) and significant class imbalance across 25 categories, leading to model overfitting and difficulty classifying underrepresented categories.
- **NLTK Setup:** Ensuring robust and consistent NLTK resource (WordNet, OMW-1.4) availability required careful local path management and manual intervention for unzipping resources correctly.
- **Iterative Model Refinement:** The best performing model and feature set were achieved through iterative experimentation with TF-IDF parameters, different classification algorithms, and hyperparameter tuning.
- **Evaluation Nuances:** This project highlighted the importance of F1-macro, weighted F1, and per-class metrics over simple accuracy, especially for imbalanced datasets, to get a true sense of model performance.

---

## 🔮 Future Work & Enhancements

- **Dataset Expansion:** Collect a significantly larger, more diverse, and balanced dataset to improve generalization and performance on minority classes.
- **Advanced NLP Features:** Explore word embeddings (Word2Vec, GloVe, FastText) or contextual embeddings from Transformers (e.g., sentence-BERT) for richer text representation.
- **Deep Learning Models:** With more data, experiment with CNNs, RNNs/LSTMs, or fine-tune pre-trained Transformer models specifically for this classification task.
- **Explainable AI (XAI):** Integrate techniques like LIME or SHAP to provide insights into model predictions, increasing transparency.
- **Enhanced Functionality:**
    - Develop functionality to rank resumes against specific job descriptions (beyond just category matching).
    - Extract specific skills, years of experience, and education levels using Named Entity Recognition (NER).
- **Bias Auditing & Mitigation:** Implement more formal tools and methodologies to detect and reduce potential biases in the model and data.
- **OCR Integration:** Add support for processing image-based PDFs to extract text.

---
