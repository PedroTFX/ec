import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv('biodegradable_a.csv')

# Handle missing values by interpolating
df.interpolate(method='linear', inplace=True, limit_direction='both')

# If there are still missing values, fill them with forward fill or backward fill
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# Define your X and y
X = df.drop(['Biodegradable'], axis=1).values
y = df['Biodegradable'].values

# Split your data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize your data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define classifiers
classifiers = {
    'Logistic Regression' : LogisticRegression(max_iter=3000),
    'Decision Tree' : DecisionTreeClassifier(),
    'KNN' : KNeighborsClassifier()
}

# Define hyperparameters
hyperparameters = {
    'Logistic Regression' : { 'C': [0.1, 1, 10, 100], 'solver': ['newton-cg', 'liblinear']},
    'Decision Tree' : {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]},
    'KNN' : {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'algorithm': ['ball_tree', 'kd_tree']}
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
    print()

# Print the best performance for each model
for name in classifiers.keys():
    model_results = [result for result in all_results if result[0] == name]
    best_result = max(model_results, key=lambda x: x[2])
    print(f"Best performance of {name}: Hyperparameters {best_result[1]}, Performance {best_result[2]}")

    # Refit the best model and evaluate it
    best_clf = classifiers[name]
    best_clf.set_params(**best_result[1])
    best_clf.fit(X_train, y_train)
    y_pred = best_clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix: \n {confusion_matrix(y_test, y_pred)}")
    print(f"Classification Report: \n {classification_report(y_test, y_pred)}")
    print()
