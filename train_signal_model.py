import pandas as pd
from sklearn.metrics import classification_report
from xgboost import XGBClassifier

def train_signal_model(df: pd.DataFrame) -> XGBClassifier:
    """
    Train a simple classifier on time-ordered signal data.
    Assumes the target column is named 'win_trade' and that
    all feature columns are numeric.
    """
    # Drop rows with missing target values
    df = df.dropna(subset=["win_trade"]).copy()

    # Fill missing feature values with column medians
    feature_cols = [col for col in df.columns if col != "win_trade"]
    for col in feature_cols:
        df[col] = df[col].fillna(df[col].median())

    # Keep the split chronological to avoid look-ahead bias
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df["win_trade"]
    X_test = test_df[feature_cols]
    y_test = test_df["win_trade"]

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

    return model
