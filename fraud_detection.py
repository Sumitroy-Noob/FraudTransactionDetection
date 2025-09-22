import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('data.csv')

# Convert TX_DATETIME to datetime objects
df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df['TX_DATETIME'] = df['TX_DATETIME'].dt.date

# Sort the DataFrame by TX_DATETIME
df = df.sort_values(by='TX_DATETIME')

# Feature Engineering
# Scenario 1: Simple amount threshold
df['AMOUNT_OVER_220'] = (df['TX_AMOUNT'] > 220).astype(int)

# Scenario 2: Rolling count of fraudulent transactions per terminal
df['TERMINAL_FRAUD_COUNT_28D'] = df.groupby('TERMINAL_ID')['TX_FRAUD'].transform(lambda x: x.rolling(window=28, closed='left').sum())
df['TERMINAL_FRAUD_COUNT_28D'] = df['TERMINAL_FRAUD_COUNT_28D'].fillna(0)

# Scenario 3: Rolling average transaction amount per customer
df['CUSTOMER_AVG_TX_AMOUNT_14D'] = df.groupby('CUSTOMER_ID')['TX_AMOUNT'].transform(lambda x: x.rolling(window=14, closed='left').mean())
df['CUSTOMER_AVG_TX_AMOUNT_14D'] = df['CUSTOMER_AVG_TX_AMOUNT_14D'].fillna(0)

# Define features (X) and target (y)
features = ['TX_AMOUNT', 'AMOUNT_OVER_220', 'TERMINAL_FRAUD_COUNT_28D', 'CUSTOMER_AVG_TX_AMOUNT_14D']
X = df[features]
y = df['TX_FRAUD']

# Chronological split of the data
# We'll use the first 80% of the data for training and the last 20% for testing
split_point = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

# Build the Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()