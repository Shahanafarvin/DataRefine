# datarefine/handle_missing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingDataHandler:
    """
    A class used to handle missing data in a DataFrame.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing missing data to be handled.
    """
    
    def __init__(self, dataframe):
        """
        Initialize the MissingDataHandler with the provided DataFrame.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be processed for missing data.
        """
        self.dataframe = dataframe

    def visualize_missing(self, filename=None):
        """
        Visualize the count of missing values in each column of the DataFrame.

        Parameters:
        ----------
        filename : str, optional
            The name of the file to save the plot. If None, the plot is shown instead.
        """
        plt.figure(figsize=(10, 6))
        missing_counts = self.dataframe.isnull().sum()
        sns.barplot(x=missing_counts.index, y=missing_counts.values)

        # Annotate each bar with the missing count
        for column_index, missing_count in enumerate(missing_counts.values):
            plt.text(column_index, missing_count, f'{missing_count}', ha='center', va='bottom')

        plt.title("Missing Values Count per Column")
        plt.xlabel("Columns")
        plt.ylabel("Count of Missing Values")
        plt.xticks(rotation=45)
        
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def impute(self, strategy='mean', fill_value=None, **kwargs):
        """
        Impute the missing values in the DataFrame using the specified strategy.

        Parameters:
        ----------
        strategy : str
            The imputation strategy to use ('mean', 'median', 'most_frequent', 'predictive', or 'custom').
        fill_value : any, optional
            The value to use for the 'custom' imputation strategy. Required if strategy is 'constant'.
        kwargs : dict
            Additional arguments to pass to the IterativeImputer for 'predictive' strategy.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'predictive':
            imputer = IterativeImputer(**kwargs)
        elif strategy == 'custom':
            if fill_value is None:
                raise ValueError("fill_value must be provided for constant imputation")
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        else:
            raise ValueError("Unsupported imputation strategy")

        # Visualize missing data before imputation
        self.visualize_missing()

        imputed_data = imputer.fit_transform(self.dataframe)
        self.dataframe = pd.DataFrame(imputed_data, columns=self.dataframe.columns)
        
        # Visualize missing data after imputation
        self.visualize_missing()
        
        return self.dataframe
    
    