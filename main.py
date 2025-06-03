import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split

if __name__ == '__main__':
    df = pd.read_csv("covid_related_disease_data.csv")

    # Check for missing values
    if df.isnull().values.any():
        missing_info = df.isnull().sum()
        print(f"Dataset contains missing values:\n{missing_info}")

    for col in df.select_dtypes(include='object'):
        df[col] = pd.factorize(df[col])[0]

    print(df)

    features = ["Age", "Symptoms", "Severity", "Smoking_Status", "BMI"]
    target = 'Hospitalized'
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    cv = cross_val_score(model, X, y, cv=10)
    print("CV Scores:", cv)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    convert = lambda x: ['yes' if val == 0 else 'no' for val in x]

    y_test_converted = convert(y_test)
    y_pred_converted = convert(y_pred)

    df = pd.DataFrame({
        "y_pred": y_pred_converted,
        "y_test": y_test_converted
    })

    df.to_csv("predictions.csv", index=False)
