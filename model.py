import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load data
df = pd.read_csv('loan_approval.csv')
#df.info()

#preprocesssing
#df['loan_approved'] = df['loan_approved'].map({True : 1, False : 0})
df['loan_approved'] = df['loan_approved'].astype(int)
df = df.dropna(subset=['loan_approved'])
#df.info()

#removed points from features as it is causing data leakage
Features = ['income', 'credit_score', 'loan_amount', 'years_employed']
Target = df['loan_approved']
X = df[Features]
y = df['loan_approved']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#model and prediction
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

#pickle to save model for streamlit app to run fast otherwise v wud hv to do the model training in the streamlit file which would slow it down too much
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved successfully!")