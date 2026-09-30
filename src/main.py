# EFWKQLFNLEWFNIWQFNWEL
from load_data import load_data
from eda import run_eda
from feature_engineering import create_features
from preprocess import preprocess_data
from train_test_split import split_data
from train import train_model, save_model
from evaluate import evaluate_model, plot_predictions
import joblib

df = load_data()

df = create_features(df)

df = df.dropna()

train_df, test_df = split_data(df)

print("Training Data:")
print(train_df.shape)
print(train_df["date"].min())
print(train_df["date"].max())

print("\nTesting Data:")
print(test_df.shape)
print(test_df["date"].min())
print(test_df["date"].max())

X_train, y_train = preprocess_data(train_df)
X_test, y_test = preprocess_data(test_df)

joblib.dump(
    X_train.columns.tolist(),
    "models/model_columns.pkl"
)

print("\nX_train:")
print(X_train.shape)

print("\nX_test:")
print(X_test.shape)

print("\nTraining Gradient Boosting model...")

model = train_model(X_train, y_train)

print("Model trained sucessfully!")

save_model(model)

print("\nEvaluating model...")

predictions = evaluate_model(
    model, X_test, y_test
)

plot_predictions(y_test, predictions)

run_eda(df)