from sklearn.ensemble import GradientBoostingRegressor
import joblib

def train_model(X_train, y_train):
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

def save_model(model, path="models/gradient_boosting_model.pkl"):
    joblib.dump(model, path)
    print(f"Model saved to {path}")