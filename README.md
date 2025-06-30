# Sentiment-Analysis-Movie-Reviews
This is a machine learning model aimed at analyzing the sentiment of IMDb movie reviews. The objective is to classify reviews as **positive** or **negative** using **TF-IDF vectorization** and **machine learning models** like Logistic Regression and Random Forest.

# Objective
To build a text classification model that identifies sentiment from movie reviews using classical machine learning techniques.

# Dataset
https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Tools and Libraries
- Scikit-learn
- Python
- SpaCy
- Jupyter Notebook
- Pandas, NumPy
- TF-IDF Vectorizer
- Matplotlib / Seaborn

# Results

- **Logistic Regression Accuracy:** ~87%
- **Random Forest Accuracy:** ~84%
- **SVC Accuracy:** ~85%
- Evaluation done using: Accuracy Score, Confusion Matrix, and F1-Score

# Team members
**Group No. 32**
- Santwana Behara(Team Leader)
- Mohammad Rakshanda
- Majji Vivek
- Shibin Malakot

# How to Run the Project

Follow these steps to run the Sentiment Analysis on Movie Reviews project:

### 1. Clone the Repository
git clone https://github.com/Shibin08/sentiment-analysis-movie-reviews.git
cd sentiment-analysis-movie-reviews

### 2. Install Required Libraries
pip install -r requirements.txt

### 3. Open the Notebook
Launch Jupyter Notebook or use VS Code to open and run the file:
Model.ipynb

# Loading the Trained Models

To load the following files in the `saved_model/` folder:

- `LogisticRegression_model.sav`
- `LinearSVC_model.sav`

### How to Use
import pickle

logistic_model = pickle.load(open("saved_model/LogisticRegression_model.sav", "rb"))
svc_model = pickle.load(open("saved_model/LinearSVC_model.sav", "rb"))

sample = ["The movie was fantastic!"]
print("Logistic Regression:", logistic_model.predict(sample)[0])
print("Linear SVC:", svc_model.predict(sample)[0])





