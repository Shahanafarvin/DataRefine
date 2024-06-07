# example_usage.py

import pandas as pd
from datarefine.handle_missing import MissingDataHandler

def main():
    """
    Main function to demonstrate the usage of the MissingDataHandler class.

    This function creates a sample DataFrame with missing values, initializes the
    MissingDataHandler with the DataFrame, visualizes missing values before and after
    imputation, and prints the DataFrame before and after imputation.
    """
    
    # Create a sample DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 2, 4],
        'C': [1, None, 2, None]
    })

    # Initialize the MissingDataHandler with the sample DataFrame
    handler = MissingDataHandler(df)

    # Visualize missing values before imputation
    handler.visualize_missing(filename="before.png")
    print("Before Imputation:\n", df)

    # Impute missing values using the 'custom' strategy with fill_value=0
    imputed_df = handler.impute(strategy='custom', fill_value=0)
    print("After Imputation:\n", imputed_df)

    # Visualize missing values after imputation
    handler.visualize_missing(filename="after.png")

if __name__ == "__main__":
    main()
