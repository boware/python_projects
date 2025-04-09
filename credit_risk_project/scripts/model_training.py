import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load the dataset with features
    df = pd.read_csv('data/feature_engineered_train.csv')

    # Separate features and target variable
    X = df.drop('TARGET', axis=1)  # 'TARGET' is the column indicating credit risk
    y = df['TARGET']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Print evaluation metrics
    print("Model Evaluation:")
    print(classification_report(y_test, y_pred))

    # Save the model for future use
    import joblib
    joblib.dump(model, 'credit_risk_model.pkl')
    print("Model saved as 'credit_risk_model.pkl'")

if __name__ == "__main__":
    train_model()
