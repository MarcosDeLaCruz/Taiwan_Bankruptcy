import pandas as pd
import pickle

def wrangle(filepath):
    """
    Reads a CSV file and prepares it for prediction.
    Assumes 'bankrupt' column was used in training and must be removed before predicting.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame ready for prediction.
    """
    df = pd.read_csv(filepath)

    # If 'bankrupt' column exists, drop it (only features are needed)
    if 'bankrupt' in df.columns:
        df = df.drop(columns='bankrupt')

    return df

def make_predictions(data_filepath, model_filepath):
    """
    Load the dataset and a trained model, return predictions.

    Parameters:
        data_filepath (str): Path to input CSV file.
        model_filepath (str): Path to the trained .pkl model.

    Returns:
        pd.Series: Predictions indexed like the input.
    """
    X = wrangle(data_filepath)

    with open(model_filepath, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X)
    return pd.Series(y_pred, index=X.index)
