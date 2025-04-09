import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def preprocess_data(df):
    # Handling missing values
    imputer = SimpleImputer(strategy='most_frequent')
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Feature scaling
    scaler = StandardScaler()
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    print("Preprocessing complete!")
    return df

def load_and_preprocess():
    # Load dataset
    df = pd.read_csv('home-credit-default-risk.csv')  

    # Preprocess data
    df = preprocess_data(df)

    # Save preprocessed data
    df.to_csv('data/preprocessed_train.csv', index=False)
    print("Preprocessed data saved to data/preprocessed_train.csv")

if __name__ == "__main__":
    load_and_preprocess()
