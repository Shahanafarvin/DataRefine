# example_usage_missing_data.py

import pandas as pd
import numpy as np
from datarefine.handle_missing import MissingDataHandler

def main():
    """
    Main function to demonstrate the usage of the MissingDataHandler class.

    This function creates a sample DataFrame with missing values, initializes the MissingDataHandler with the DataFrame,
    imputes the missing values, visualizes the missing data before and after imputation,
    and prints the DataFrame before and after processing.
    """

    # Create a sample DataFrame with missing values
    df = pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [np.nan, 2, 2, 4],
        'C': [1, np.nan, 2, np.nan]
    })

    # Initialize the MissingDataHandler with the sample DataFrame
    handler = MissingDataHandler(df)

    # Visualize missing data before imputation
    handler.visualize(plot_type='heatmap', title="Missing Values Before Imputation (Heatmap)")
    handler.visualize(plot_type='bar', title="Missing Values Before Imputation (Bar)")

    # Impute the missing values using the mean
    imputed_df = handler.impute_missing(method='mean')
    print("DataFrame After Mean Imputation:\n", imputed_df)

    # Initialize the MissingDataHandler with the imputed DataFrame
    handler_imputed = MissingDataHandler(imputed_df)

    # Visualize missing data after imputation
    handler_imputed.visualize(plot_type='heatmap', title="Missing Values After Imputation (Heatmap)")
    handler_imputed.visualize(plot_type='bar', title="Missing Values After Imputation (Bar)")

if __name__ == "__main__":
    main()
