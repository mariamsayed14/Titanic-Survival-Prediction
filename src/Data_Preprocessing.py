import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def read_data():
    train = pd.read_csv("../data/Titanic_train.csv")
    return train


# Simple Cleaning
def simplify_columns(df):
    df = df.copy()

    # standardize text (remove spaces + lowercase)
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].str.lower().str.strip()

    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].str.upper().str.strip()

    return df


# Handle Missing Values
def fill_missing_values(df):
    df = df.copy()

    #Fill Embarked with Mode (categorical)
    if 'Embarked' in df.columns:
        most_common = df['Embarked'].mode()[0]
        df['Embarked'] = df['Embarked'].fillna(most_common)

    #Fill Age with Median (Numerical)
    if 'Age' in df.columns:
        median_age = df['Age'].median()
        df['Age'] = df['Age'].fillna(median_age)
    return df


# Encoding
def encode_data(df):
    df = df.copy()

    # convert Sex to numbers
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # apply one-hot encoding to categorical columns
    cat_cols = []
    if 'Embarked' in df.columns:
        cat_cols.append('Embarked')
    if 'Pclass' in df.columns:
        cat_cols.append('Pclass')
    df = pd.get_dummies(df, columns=cat_cols,drop_first=True)

    #Feature Engineering - Joining SibSp and Parch into FamilySize
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

    return df


#Drop Columns
def drop_columns(df):
    df=df.copy()

    if 'Name' in df.columns:
        df.drop('Name', axis=1, inplace=True)
    if 'Ticket' in df.columns:
        df.drop('Ticket', axis=1, inplace=True)
    if 'Cabin' in df.columns:
        df.drop('Cabin', axis=1, inplace=True)
    if 'PassengerId' in df.columns:
        df.drop('PassengerId', axis=1, inplace=True)
    if 'SibSp' in df.columns:
        df.drop('SibSp', axis=1, inplace=True)
    if 'Parch' in df.columns:
        df.drop('Parch', axis=1, inplace=True)

    return df



def handle_outliers(df):
    df = df.copy()
    for col in ['Fare', 'Age']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Instead of dropping, clip the values to the bounds
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

# Full preprocessing
def preprocess(df):
    df = simplify_columns(df)
    df = handle_outliers(df)
    df = fill_missing_values(df)
    df = encode_data(df)
    df = drop_columns(df)

    return df


#Scaling
def scale_columns(df , df_test):
    scaler = MinMaxScaler()
    cols_to_scale = []

    if 'Age' in df.columns:
        cols_to_scale.append('Age')
    if 'Fare' in df.columns:
        cols_to_scale.append('Fare')

    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    df_test[cols_to_scale] = scaler.transform(df_test[cols_to_scale])

    return df , df_test


# separate target
def separate_target(df,target):
    y = df[target]
    x = df.drop(target, axis=1)
    return x,y

if __name__ == "__main__":
    train =read_data()

    X,Y=separate_target(train,'Survived')
    # preprocess data
    X = preprocess(X)
    print("Train shape:", X.shape)
    print("\nTrain Columns:")
    print(X.columns)
    print("\nSample:")
    print(X.head())