""" import pandas as pd
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
print("Classification Report:\n", classification_report(y_test, y_pred)) """

""" import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers and their respective parameter grids
classifiers = [
    {
        'clf': RandomForestClassifier(),
        'params': {
            'n_estimators': [10, 50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        'clf': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 1, 10],
            'kernel': ['linear', 'rbf']
        }
    },
    {
        'clf': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    }
]

# Loop over the classifiers
for classifier in classifiers:
    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=classifier['clf'], param_grid=classifier['params'], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters for", type(classifier['clf']).__name__, ":", best_params)
    print("Best Cross-Validation Score for", type(classifier['clf']).__name__, ":", best_score)

    # Train the classifier with the best parameters on the training data
    best_clf = classifier['clf'].set_params(**best_params)
    best_clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = best_clf.predict(X_test)

    # Evaluate the classifier's performance
    print("Test Accuracy for", type(classifier['clf']).__name__, ":", accuracy_score(y_test, y_pred))
    print("Confusion Matrix for", type(classifier['clf']).__name__, ":\n", confusion_matrix(y_test, y_pred))
    print("Classification Report for", type(classifier['clf']).__name__, ":\n", classification_report(y_test, y_pred)) """

""" import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer

# Load the dataset
data = pd.read_csv('biodegradable_a.csv')

# Separate features and target columns
X = data.drop('Biodegradable', axis=1)  # Replace 'Biodegradable' with the name of the target column
y = data['Biodegradable']

# Handle missing values for feature columns using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X.iloc[:, :] = imputer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers and their respective parameter grids
classifiers = [
    {
        'clf': LogisticRegression(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
    },
    {
        'clf': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        'clf': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    }
]

# Loop over the classifiers
for classifier in classifiers:
    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=classifier['clf'], param_grid=classifier['params'], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters for", type(classifier['clf']).__name__, ":", best_params)
    print("Best Cross-Validation Score for", type(classifier['clf']).__name__, ":", best_score)

    # Train the classifier with the best parameters on the training data
    best_clf = classifier['clf'].set_params(**best_params)
    best_clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = best_clf.predict(X_test)

    # Evaluate the classifier's performance
    print("Test Accuracy for", type(classifier['clf']).__name__, ":", accuracy_score(y_test, y_pred))
    print("Confusion Matrix for", type(classifier['clf']).__name__, ":\n", confusion_matrix(y_test, y_pred))
    print("Classification Report for", type(classifier['clf']).__name__, ":\n", classification_report(y_test, y_pred)) """

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import KNNImputer
import warnings

# Suppress FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Load the dataset
data = pd.read_csv('biodegradable_a.csv')

# Separate features and target columns
X = data.drop('Biodegradable', axis=1)  # Replace 'target_column' with the name of the target column
y = data['Biodegradable']

# Handle missing values for feature columns using KNNImputer
imputer = KNNImputer(n_neighbors=5)
X.iloc[:, :] = imputer.fit_transform(X)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers and their respective parameter grids
classifiers = [
    {
        'clf': LogisticRegression(max_iter=3000),
        'params': {
            'C': [0.1, 1, 10, 100],
            'solver': ['newton-cg', 'liblinear']
        }
    },
    {
        'clf': DecisionTreeClassifier(),
        'params': {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    },
    {
        'clf': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['ball_tree', 'kd_tree']
        }
    }
]

# Loop over the classifiers
for classifier in classifiers:
    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(estimator=classifier['clf'], param_grid=classifier['params'], cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best parameters and the corresponding score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best Parameters for", type(classifier['clf']).__name__, ":", best_params)
    print("Best Cross-Validation Score for", type(classifier['clf']).__name__, ":", best_score)

    # Train the classifier with the best parameters on the training data
    best_clf = classifier['clf'].set_params(**best_params)
    best_clf.fit(X_train, y_train)

    # Predict the labels for the test data
    y_pred = best_clf.predict(X_test)

    # Evaluate the classifier's performance
    print("Test Accuracy for", type(classifier['clf']).__name__, ":", accuracy_score(y_test, y_pred))
    print("Confusion Matrix for", type(classifier['clf']).__name__, ":\n", confusion_matrix(y_test, y_pred))
    print("Classification Report for", type(classifier['clf']).__name__, ":\n", classification_report(y_test, y_pred))
