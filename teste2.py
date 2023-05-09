import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer

# Load the dataset
data = pd.read_csv('biodegradable_a.csv')

# Separate features and target columns
X = data.drop('Biodegradable', axis=1)  # Replace 'target_column' with the name of the target column
y = data['Biodegradable']

# Handle missing values for feature columns using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X.iloc[:, :] = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Choose and import the classifier
clf = RandomForestClassifier()
# Define the hyperparameter search space
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search with Cross-Validation
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters and the corresponding score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)

# Train the classifier with the best parameters on the training data
best_clf = RandomForestClassifier(**best_params)
best_clf.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = best_clf.predict(X_test)

# Evaluate the classifier's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
