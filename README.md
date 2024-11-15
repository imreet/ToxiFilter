# Content Moderation System
This project implements a machine learning-based content moderation system to classify comments as toxic or non-toxic using logistic regression. It uses text data, processes it with TF-IDF, and incorporates SMOTE for handling class imbalance.

# Features
Text classification for toxic comment detection.
Handles class imbalance using SMOTE.
Combines TF-IDF vectorization and additional features like comment length.
Logistic regression model optimized using RandomizedSearchCV.
Achieves an accuracy of approximately 90% - 95%.

# Requirements
Before running the code, ensure you have the following Python packages installed:
1. pandas
2. scikit-learn
3. imblearn
4. scipy
5. numpy
You can install these dependencies using:
pip install pandas scikit-learn imbalanced-learn scipy numpy

# How to Run
1. Clone this repository:
git clone https://github.com/your_username/content-moderation-system.git
cd content-moderation-system

2 .Place your dataset (e.g., Copy.csv) in the project directory.

3. Open the CMS.py file and ensure the dataset path is correct:
df = pd.read_csv(r'E:\Reetam\MITWPU\TY\SEM 5\AAI\MP\archive\Copy.csv', low_memory=False)

4.Run the script:
python CMS.py

5. After execution, the script will:
Print the accuracy and classification report.
Save predictions to a CSV file:
C:\ThisPC\content_moderation_predictions.csv

# Dataset Details
Input: A CSV file containing columns:
comment_text: The comment to analyze.
toxic_score: The toxicity score used to label the data.
Labels:
1: Toxic comment (toxic_score > 0.6).
0: Non-toxic comment.

# Performance
Accuracy: ~90%-95% (can vary depending on dataset size and hyperparameter tuning).
Precision, Recall, and F1-Score: Refer to the classification report printed after execution.

# Customization
1. Adjust the sample() function in the script to reduce dataset size for faster prototyping:
   df = df.sample(frac=0.5, random_state=22)
2. Modify param_dist in RandomizedSearchCV to test different hyperparameters:
  param_dist = {
      'C': [0.01, 0.1, 1, 10, 100],
      'solver': ['lbfgs', 'liblinear']
   }

# License
This project is licensed under the MIT License. Feel free to use and modify it as needed.
