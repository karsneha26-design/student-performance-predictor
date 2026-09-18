import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Create sample student-performance data
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                    2, 4, 6, 8, 3, 5, 7, 9, 1, 10],
    
    "attendance": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                   52, 68, 78, 88, 62, 72, 82, 92, 48, 98],
    
    "previous_marks": [40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
                       42, 58, 68, 78, 52, 63, 73, 83, 38, 94],
    
    "assignments": [2, 3, 4, 5, 6, 7, 8, 9, 10, 10,
                    3, 5, 7, 9, 4, 6, 8, 10, 2, 10],
    
    "performance": [42, 48, 53, 58, 64, 69, 75, 81, 87, 94,
                    45, 59, 72, 84, 55, 67, 77, 89, 40, 97]
}

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Input features
X = df[["study_hours", "attendance", "previous_marks", "assignments"]]

# Target value
y = df["performance"]

# Create and train the ML model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save the trained model
joblib.dump(model, "student_performance_model.pkl")

print("Model trained successfully!")
print("Model saved as student_performance_model.pkl")