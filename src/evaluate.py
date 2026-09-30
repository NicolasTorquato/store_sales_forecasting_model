from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print("----------------")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.2f}")

    return predictions

def plot_predictions(y_test, predictions):
    plt.figure(figsize=(10,6))

    plt.scatter(y_test, predictions, alpha=0.3)

    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.title("Actual vs Predicted Sales")

    # Perfect Prediction line
    max_value = max(y_test.max(), predictions.max())

    plt.plot(
        [0, max_value], 
        [0, max_value], 
        linestyle="--"
    )

    plt.tight_layout()
    plt.show()