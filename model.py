import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# LOAD DATASET

df = pd.read_csv('dataset/philippine_gwa_dataset.csv')

# FEATURES
features = [
    'GWA',
    'coding_score',
    'communication_score',
    'aptitude_score',
    'resume_score',
    'skill_score',
    'projects',
    'internships'
]

X = df[features]

# SALARY TARGET
salary_y = df['Salary_PHP_Per_Month']

# PLACEMENT TARGET
placement_y = df['placed']

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    salary_y,
    test_size=0.2,
    random_state=42
)

# SALARY MODEL
salary_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

salary_model.fit(X_train, y_train)

joblib.dump(salary_model, 'models/salary_model.pkl')

# PLACEMENT MODEL
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X,
    placement_y,
    test_size=0.2,
    random_state=42
)

placement_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

placement_model.fit(X_train2, y_train2)

joblib.dump(placement_model, 'models/placement_model.pkl')

print('Models trained successfully!')