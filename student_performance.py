
# Student Performance Prediction
# Machine Learning Project

# 1. Import Libraries
import csv

import joblib
import matplotlib.pyplot as plt
import numpy as np


# 2. Helper functions

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def train_linear_regression(X, y):
    design_matrix = np.column_stack([np.ones(len(X)), X])
    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, y, rcond=None)
    return coefficients


def predict_linear_regression(X, coefficients):
    design_matrix = np.column_stack([np.ones(len(X)), X])
    return design_matrix @ coefficients


def train_random_forest(X, y, n_estimators=50, max_features=None, random_seed=42):
    rng = np.random.default_rng(random_seed)
    trees = []
    feature_count = X.shape[1]
    if max_features is None:
        max_features = max(1, feature_count)

    for _ in range(n_estimators):
        indices = rng.integers(0, len(X), size=len(X))
        selected_features = rng.choice(feature_count, size=max_features, replace=False)
        sampled_X = X[indices][:, selected_features]
        sampled_y = y[indices]
        coefficients = train_linear_regression(sampled_X, sampled_y)
        trees.append((selected_features, coefficients))

    return trees


def predict_random_forest(X, trees):
    predictions = np.zeros((len(X), len(trees)))
    for index, (selected_features, coefficients) in enumerate(trees):
        sampled_X = X[:, selected_features]
        predictions[:, index] = predict_linear_regression(sampled_X, coefficients)
    return predictions.mean(axis=1)


def load_student_data(file_path):
    with open(file_path, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    if not rows:
        raise ValueError("Dataset is empty.")

    columns = reader.fieldnames
    data = {
        column: np.array([float(row.get(column, np.nan)) for row in rows], dtype=float)
        for column in columns
    }
    return data, rows


data, rows = load_student_data("student_data.csv")

print("First 5 Records:")
for index in range(min(5, len(rows))):
    print({key: value for key, value in rows[index].items()})

# 3. Check Dataset
print("\nDataset Information:")
print(f"Rows: {len(rows)}")
print(f"Columns: {list(data.keys())}")

print("\nMissing Values:")
for column, values in data.items():
    missing_count = int(np.isnan(values).sum())
    if missing_count:
        print(f"{column}: {missing_count}")

# Handle missing numerical values
for column, values in data.items():
    if np.isnan(values).any():
        valid = values[~np.isnan(values)]
        median = np.median(valid) if valid.size else 0.0
        data[column] = np.where(np.isnan(values), median, values)

# 4. Data Visualization
feature_names = ["Study Hours", "Attendance", "Previous Marks", "Sleep Hours"]
feature_matrix = np.column_stack([data[name] for name in feature_names])
correlation = np.corrcoef(feature_matrix, rowvar=False)

plt.figure(figsize=(8, 5))
plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(feature_names)), feature_names, rotation=45)
plt.yticks(range(len(feature_names)), feature_names)
for row_index in range(correlation.shape[0]):
    for col_index in range(correlation.shape[1]):
        plt.text(col_index, row_index, f"{correlation[row_index, col_index]:.2f}", ha="center", va="center", color="black")
plt.title("Student Performance Correlation")
plt.colorbar(label="Correlation")
plt.show()

# 5. Select Features and Target
X = np.column_stack([data[name] for name in feature_names])
y = data["Marks"]

# 6. Split Dataset
rng = np.random.default_rng(42)
indices = np.arange(len(X))
rng.shuffle(indices)
split_index = int(len(X) * 0.8)
X_train = X[indices[:split_index]]
X_test = X[indices[split_index:]]
y_train = y[indices[:split_index]]
y_test = y[indices[split_index:]]

# 7. Initialize Models
models = {
    "Linear Regression": (train_linear_regression, predict_linear_regression),
    "Random Forest": (train_random_forest, predict_random_forest),
}

# 8. Train and Evaluate Models
results = {}

for name, (train_fn, predict_fn) in models.items():
    if name == "Linear Regression":
        model = train_fn(X_train, y_train)
        predictions = predict_fn(X_test, model)
    else:
        model = train_fn(X_train, y_train, n_estimators=50, max_features=2, random_seed=42)
        predictions = predict_fn(X_test, model)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results[name] = r2

    print("\nModel:", name)
    print("MAE:", mae)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R2 Score:", r2)

# 9. Compare Models
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values())
plt.title("Model Performance Comparison")
plt.ylabel("R2 Score")
plt.show()

# 10. Select Best Model
best_model_name = max(results, key=results.get)
if best_model_name == "Linear Regression":
    best_model = (train_linear_regression, predict_linear_regression)
    best_model_params = train_linear_regression(X, y)
else:
    best_model = (train_random_forest, predict_random_forest)
    best_model_params = train_random_forest(X, y, n_estimators=50, max_features=2, random_seed=42)

print("\nBest Model:", best_model_name)

# 11. Save Model
joblib.dump({"model": best_model_name, "parameters": best_model_params}, "student_model.pkl")

print("Model saved successfully!")

# 12. Predict Student Marks
new_student = np.array([[6.0, 85.0, 75.0, 8.0]])
if best_model_name == "Linear Regression":
    predicted_marks = predict_linear_regression(new_student, best_model_params)
else:
    predicted_marks = predict_random_forest(new_student, best_model_params)

print("\nPredicted Student Marks:", round(predicted_marks[0], 2))
