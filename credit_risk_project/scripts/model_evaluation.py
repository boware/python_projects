import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_model():
    # Load the saved model
    model = joblib.load('credit_risk_model.pkl')

    # Load the testing data
    df = pd.read_csv('data/feature_engineered_train.csv')
    X_test = df.drop('TARGET', axis=1)
    y_test = df['TARGET']

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Model Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{cm}")

if __name__ == "__main__":
    evaluate_model()
