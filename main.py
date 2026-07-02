import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Read the dataset
df = pd.read_csv("data/manufacturing_defects.csv")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns)

print("\n========== DATA INFORMATION ==========")
print(df.info())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

print("\n========== DEFECT TYPE COUNT ==========")
print(df["Defect_Type"].value_counts())

# Bar Chart
df["Defect_Type"].value_counts().plot(kind="bar")

plt.title("Defect Type Distribution")
plt.xlabel("Defect Type")
plt.ylabel("Count")

plt.show()
print("\n========== PRODUCT COUNT ==========")
print(df["Product"].value_counts())

df["Product"].value_counts().plot(kind="bar")

plt.title("Product Distribution")
plt.xlabel("Product")
plt.ylabel("Count")
plt.xticks(rotation=45)

plt.show()
print("\n========== MACHINE COUNT ==========")
print(df["Machine"].value_counts())

df["Machine"].value_counts().plot(kind="bar")

plt.title("Machine-wise Inspection Count")
plt.xlabel("Machine")
plt.ylabel("Number of Inspections")

plt.show()
print("\n========== SHIFT COUNT ==========")
print(df["Shift"].value_counts())

df["Shift"].value_counts().plot(kind="bar")

plt.title("Shift-wise Inspection Count")
plt.xlabel("Shift")
plt.ylabel("Number of Inspections")

plt.show()
print("\n========== PRODUCTION LINE COUNT ==========")
print(df["Production_Line"].value_counts())

df["Production_Line"].value_counts().plot(kind="bar")

plt.title("Production Line-wise Inspection Count")
plt.xlabel("Production Line")
plt.ylabel("Number of Inspections")

plt.show()
print("\n========== SEVERITY COUNT ==========")
print(df["Severity"].value_counts())

df["Severity"].value_counts().plot(kind="bar")

plt.title("Severity Distribution")
plt.xlabel("Severity")
plt.ylabel("Number of Defects")

plt.show()
print("\n========== INSPECTION RESULT COUNT ==========")
print(df["Inspection_Result"].value_counts())

df["Inspection_Result"].value_counts().plot(kind="bar")

plt.title("Inspection Result Distribution")
plt.xlabel("Inspection Result")
plt.ylabel("Number of Products")

plt.show()
print("\n========== MAINTENANCE STATUS COUNT ==========")
print(df["Maintenance_Status"].value_counts())

df["Maintenance_Status"].value_counts().plot(kind="bar")

plt.title("Maintenance Status Distribution")
plt.xlabel("Maintenance Status")
plt.ylabel("Number of Records")

plt.show()
print("\n========== TEMPERATURE SUMMARY ==========")
print(df["Temperature"].describe())

df["Temperature"].plot(kind="hist", bins=10)

plt.title("Temperature Distribution")
plt.xlabel("Temperature (°C)")
plt.ylabel("Frequency")

plt.show()
print("\n========== AI RISK SCORE SUMMARY ==========")
print(df["AI_Risk_Score"].describe())

df["AI_Risk_Score"].plot(kind="hist", bins=10)

plt.title("AI Risk Score Distribution")
plt.xlabel("AI Risk Score")
plt.ylabel("Frequency")

plt.show()
print("\n========== LABEL ENCODING ==========")

# Create LabelEncoder object
# Show original Severity values
print("\nOriginal Severity Values:")
print(sorted(df["Severity"].unique()))

# Create LabelEncoder object
label_encoder = LabelEncoder()

# Columns to encode
categorical_columns = [
    "Product",
    "Production_Line",
    "Machine",
    "Shift",
    "Operator",
    "Material_Type",
    "Defect_Type",
    "Severity",
    "Inspection_Result",
    "Maintenance_Status"
]

# Encode each categorical column
# Encode each categorical column
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

    if column == "Severity":
        print("\nSeverity Mapping:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"{i} -> {label}")

print(df.head())
print("\n========== FEATURES AND TARGET ==========")

# Features (Input)
X = X = df[[
    "Temperature",
    "Humidity",
    "Voltage",
    "Current",
    "AI_Risk_Score"
]]

y = df["Severity"]

# Target (Output)
y = df["Severity"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)
print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)
print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)
print("\n========== TRAINING RANDOM FOREST MODEL ==========")

# Create the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

print("Model training completed successfully!")
print("\n========== MODEL PREDICTIONS ==========")

# Predict on test data
y_pred = model.predict(X_test)

print("First 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test.values[:10])
print("\n========== MODEL ACCURACY ==========")

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))
print("\n========== SAVING MODEL ==========")

joblib.dump(model, "defect_prediction_model.pkl")

print("Model saved successfully!")