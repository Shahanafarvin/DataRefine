# example_usage for handling missing value

from datarefine.handle_missing import MissingDataHandler
from datarefine.handle_outliers import OutlierHandler
import pandas as pd
"""
if __name__ == "__main__":
    df = pd.DataFrame({
        'A': [1, 2, None, None],
        'B': [None, 2, 2, 4],
        'C': [1, None, 2, None]
    })
    handler = MissingDataHandler(df)
    handler.visualize_missing("missing_before_imputation.png")
    print("Before Imputation:\n", df)
    imputed_df = handler.impute(strategy='mean')
    print("After Imputation:\n", imputed_df)
    handler.visualize_missing("missing_after_imputation.png")
    """
    
    # Example usage for handling outiers
if __name__ == "__main__":
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 100],
        'B': [10, 20, 30, 40, 50, 200]
    })
    handler = OutlierHandler(df)
    print("Before handling outliers:\n", df)
    filtered_data, outliers = handler.handle_outliers('A', method='iqr')
    print("Filtered data:\n", filtered_data)
    print("Outliers:\n", outliers)

