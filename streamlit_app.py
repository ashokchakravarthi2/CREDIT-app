import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib

# Load and preprocess the dataset
df = pd.read_csv("C:\\Users\\ASHOK\\Desktop\\creditcard.csv")
df = df.drop(columns=['Time'], axis=1)

X = df.drop(columns=['Class'], axis=1)
y = df.Class

# Split the data into training and testing sets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=.3, random_state=123)

# Scale the 'Amount' column on the training set
scaler = StandardScaler()
train_x['Amount'] = scaler.fit_transform(train_x['Amount'].values.reshape(-1, 1))

# Scale the 'Amount' column on the testing set
test_x['Amount'] = scaler.fit_transform(test_x['Amount'].values.reshape(-1, 1))

# Apply SMOTE to handle class imbalance
smote = SMOTE()
train_x, train_y = smote.fit_resample(train_x, train_y)

# Train the AdaBoost classifier
ada_boost = AdaBoostClassifier(n_estimators=100, random_state=123)
ada_boost.fit(train_x, train_y)

# Save the trained model
model_path = 'ada_boost_model.pkl'
joblib.dump(ada_boost, model_path)

# Define the Streamlit app
def main():
    st.title('Credit Card Fraud Detection')
    
    # Prediction section
    st.header('Predict Fraudulent Transactions')
    
    # Form for user input
    amount = st.number_input('Transaction Amount', value=0.0)
    features = {'Amount': amount}
    
    # Scale the 'Amount' column
    input_data = pd.DataFrame([features], dtype=float)
    input_data['Amount'] = scaler.transform(input_data['Amount'].values.reshape(-1, 1))
    
    # Make prediction
    if st.button('Predict'):
        prediction = ada_boost.predict(input_data)
        st.write(f"Predicted Class: {prediction[0]}")

if __name__ == '__main__':
    main()
