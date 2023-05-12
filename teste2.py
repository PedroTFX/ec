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

""" import pandas as pd
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
    print("Classification Report for", type(classifier['clf']).__name__, ":\n", classification_report(y_test, y_pred)) """




import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('biodegradable_a.csv')

# Handle missing values by interpolating
df.interpolate(method='linear', inplace=True, limit_direction='both')

# If there are still missing values, fill them with forward fill or backward fill
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Define your X and y
X = df.drop(['Biodegradable'], axis=1).values
y = df['Biodegradable'].values

# Split your data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize your data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Logistic Regression' : LogisticRegression(),
    'Decision Tree' : DecisionTreeClassifier(),
    'KNN' : KNeighborsClassifier()
}

# Define hyperparameters
hyperparameters = {
    'Logistic Regression' : {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    'Decision Tree' : {'max_depth': [1, 10, 20, None]},
    'KNN' : {'n_neighbors': [1, 5, 10, 20]}
}

# Store all results
all_results = []

# Iterate over classifiers and hyperparameters
for name in classifiers.keys():
    clf = classifiers[name]
    params = hyperparameters[name]
    grid_search = GridSearchCV(clf, params, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    for i in range(len(grid_search.cv_results_['params'])):
        combination = grid_search.cv_results_['params'][i]
        mean_test_score = grid_search.cv_results_['mean_test_score'][i]
        all_results.append((name, combination, mean_test_score))
        print(f"Model: {name}, Hyperparameters: {combination}, Performance: {mean_test_score}")
        clf.set_params(**combination)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        print("Accuracy: ", accuracy_score(y_val, y_pred))
        print("Confusion Matrix: \n", confusion_matrix(y_val, y_pred))
        print("Classification Report: \n", classification_report(y_val, y_pred))
    print()

# Print the best performance for each model
for name in classifiers.keys():
    model_results = [result for result in all_results if result[0] == name]
    best_result = max(model_results, key=lambda x: x[2])
    print(f"Best performance of {name}: Hyperparameters {best_result[1]}, Performance {best_result[2]}")
    # Fit the model with the best hyperparameters on the combined train and validation sets,
    # and evaluate on the test set
    clf.set_params(**best_result[1])
    X_train_val = np.concatenate((X_train, X_val), axis=0)
    y_train_val = np.concatenate((y_train, y_val), axis=0)
    clf.fit(X_train_val, y_train_val)
    y_pred = clf.predict(X_test)
    print("Test Accuracy: ", accuracy_score(y_test, y_pred))
    print("Test Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
    print("Test Classification Report: \n", classification_report(y_test, y_pred))
    print()

