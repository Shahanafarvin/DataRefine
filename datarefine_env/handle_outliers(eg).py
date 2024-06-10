# example_usage_outliers.py

import pandas as pd
from datarefine.handle_outliers import OutlierHandler

def main():
    """
    Main function to demonstrate the usage of the OutlierHandler class.

    This function creates a sample DataFrame with outliers, initializes the
    OutlierHandler with the DataFrame, visualizes outliers before and after
    handling, and prints the DataFrame before and after handling outliers.
    """

    # Create a sample DataFrame with outliers
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 100],
        'B': [5, 6, 7, 8, 200],
        'C': [9, 10, 11, 12, 300]
    })

    # Initialize the OutlierHandler with the sample DataFrame
    handler = OutlierHandler(df)

    # Visualize outliers before handling
    handler.visualize_outliers(title="Outliers Before Handling",filename="befor_handling_outliers.png")
    print("Before Handling Outliers:\n", df)

    # Handle outliers using the 'remove' method with 'zscore' detection method
    cleaned_df = handler.handle_outliers(method='remove', detection_method='zscore', threshold=1.0)
    print("After Handling Outliers (Remove Method):\n", cleaned_df)

    # Visualize outliers after handling
    handler.visualize_outliers(title="Outliers After Handling",filename="after_handling_outliers.png")

if __name__ == "__main__":
    main()
