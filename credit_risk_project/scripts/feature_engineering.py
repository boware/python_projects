import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def feature_engineering(df):
    # Example feature: "loan_to_income"
    df['loan_to_income'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']

    # Handling categorical variables
    encoder = OneHotEncoder(drop='first', sparse=False)
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        encoded = encoder.fit_transform(df[[col]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
        df = pd.concat([df, encoded_df], axis=1)
        df.drop(columns=[col], inplace=True)

    print("Feature Engineering complete!")
    return df

def create_features():
    # Load preprocessed dataset
    df = pd.read_csv('data/preprocessed_train.csv')

    # Apply feature engineering
    df = feature_engineering(df)

    # Save the data with new features
    df.to_csv('data/feature_engineered_train.csv', index=False)
    print("Feature-engineered data saved to data/feature_engineered_train.csv")

if __name__ == "__main__":
    create_features()
