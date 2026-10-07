import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load dataset (update path if necessary)
df = pd.read_excel(r"C:\Users\user\Downloads\Credit_worthy\CreditWorthiness.xlsx")
df.columns = df.columns.str.capitalize()

# 2. Preprocessing & Encoding (matching notebook pipeline)
df_encoded = pd.get_dummies(df, drop_first=True)
target_col = df_encoded.columns[-1]

X = df_encoded.drop(target_col, axis=1)
y = df_encoded[target_col]

# Save exact feature columns for input alignment in Streamlit
joblib.dump(list(X.columns), 'model_columns.joblib')

# 3. Train-Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Train Model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# 5. Save Model and Scaler
joblib.dump(model, 'credit_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
print('Model training complete! Artifacts saved successfully.')