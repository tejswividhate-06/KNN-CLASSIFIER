import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump

# Load dataset
data = pd.read_csv("bc.csv",sep="\t")

# Remove spaces from column names
data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

print("Columns in dataset:")
print(data.columns.tolist())

# Remove ID column if present
if "id" in data.columns:
    data = data.drop("id", axis=1)

# Input features
X = data.drop("diagnosis", axis=1)

# Target
y = data["diagnosis"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create KNN model
model = KNeighborsClassifier(n_neighbors=5)

# Train model
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print("Model Accuracy:", accuracy)

# Save model and scaler
dump((model, scaler), "KNNModel.joblib")

print("KNNModel.joblib created successfully!")