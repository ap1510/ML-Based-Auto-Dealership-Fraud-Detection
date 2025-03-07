import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("Title.csv")

# Convert 'Title' column to binary labels
df["Title"] = df["Title"].apply(lambda x: 1 if x == 1 else 0)

df = df.set_index("Sn")

# Balance dataset using stratified sampling
pos_samples = df[df["Title"] == 1]
neg_samples = df[df["Title"] == 0].sample(n=len(pos_samples), random_state=42)
data_balanced = pd.concat([pos_samples, neg_samples]).sample(frac=1, random_state=42)

# Load TF-IDF features
tfidf_df = pd.read_csv("tfidfneg_max20.csv").set_index("Index")
X = tfidf_df.loc[data_balanced.index].to_numpy()
y = data_balanced['Title'].to_numpy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# Train SVM with GridSearchCV
param_grid = {'C': [1, 10, 100, 1000], 'gamma': [1, 0.1, 0.001, 0.0001], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(probability=True), param_grid, cv=5, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# Model Evaluation
y_pred = grid.predict(X_test)
print("Best Parameters:", grid.best_params_)
print("F1 Score:", f1_score(y_test, y_pred, average='macro'))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    grid.fit(X_train, y_train)
    print("Cross-validation Best Params:", grid.best_params_)
    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Apply model to Texas dataset
texas_data = pd.read_csv("Texas_Neg_Tf-idf_without_threshold.csv").set_index("Unnamed: 0")
pred_texas = grid.predict(texas_data.to_numpy())

# Save results
res_texas = pd.DataFrame({"Label": pred_texas}, index=texas_data.index)
res_texas["Label"].value_counts()
filtered_results = res_texas[res_texas["Label"] == 1]
filtered_results.to_csv("Title_result.csv")

# Save trained model
pickle.dump(grid, open("Title.sav", 'wb'))

print("Model training and evaluation complete. Results saved.")
