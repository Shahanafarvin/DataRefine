# example_usage_missing.py

import pandas as pd
from datarefine.handle_missing import MissingDataHandler

def main():
    """
    Main function to demonstrate the usage of the MissingDataHandler class.

    This function creates a sample DataFrame with missing values in both text and number columns,
    initializes the MissingDataHandler with the DataFrame, imputes the missing values,
    visualizes the missing data before and after imputation, and prints the DataFrame before and after processing.
    """
    # Create a sample DataFrame with text and number columns including missing values
    df = pd.DataFrame({
        'A': [1, 2, None, 4],
        'B': [None, 2, 2, 4],
        'C': [1, None, 2, None],
        'D': ['apple', 'banana', None, 'cherry']
    })

    print("Before Imputation:\n", df)

    # Initialize the MissingDataHandler with the sample DataFrame
    handler = MissingDataHandler(df)

    # Visualize missing data before imputation
    handler.visualize_missing(plot_type='bar', title="Missing Values Before Imputation", filename="missing_before.png")

    # Impute the missing values using mean strategy for numerical columns
    # and most_frequent strategy for text columns
    numerical_columns = df.select_dtypes(include=['number']).columns
    text_columns = df.select_dtypes(include=['object']).columns

    # Impute numerical columns using mean strategy
    df[numerical_columns] = handler.impute_missing(strategy='custom', dataframe=df[numerical_columns],fill_value=0)

    # Update handler with the modified dataframe for text column imputation
    handler.dataframe = df

    # Impute text columns using most_frequent strategy
    df[text_columns] = handler.impute_missing(strategy='most_frequent', dataframe=df[text_columns])

    # Visualize missing data after imputation
    handler.visualize_missing(plot_type='bar', title="Missing Values After Imputation", filename="missing_after.png")

    print("After Imputation:\n", df)

if __name__ == "__main__":
    main()
