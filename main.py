from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

if __name__ == '__main__':
    df = pd.read_csv("covid_related_disease_data.csv")

    for col in df.select_dtypes(include='object'):
        df[col] = pd.factorize(df[col])[0]

    print(df)

    features = ["Age", "Symptoms", "Severity", "Smoking_Status", "BMI", "Preexisting_Condition", "Hospitalized", "Vaccination_Status"]
    target = 'Reinfection'
    x = df[features]
    y = df[target]
    x_train = x[:500]
    y_train = y[:500]
    x_test = x[500:]
    y_test = y[500:]

    model = RandomForestClassifier()
    cv = cross_val_score(model, x, y, cv=10)
    print("CV Scores:", cv)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    convert = lambda x: ['no' if val == 0 else 'yes' for val in x]

    y_test_converted = convert(y_test)
    y_pred_converted = convert(y_pred)

    df = pd.DataFrame({
        "y_pred": y_pred_converted,
        "y_test": y_test_converted
    })

    df.to_csv("predictions.csv", index=False)
