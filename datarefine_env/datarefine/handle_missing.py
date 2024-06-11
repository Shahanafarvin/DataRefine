# datarefine/handle_missing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno
import numpy as np
import os

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

    def visualize_missing(self, plot_type='heatmap', title="Missing Values", filename=None):
        """
        Visualize the missing values in the DataFrame using the specified plot type.

        Parameters:
        ----------
        plot_type : str
            The type of plot to use ('heatmap', 'bar', 'matrix').
        title : str
            The title of the plot.
        filename : str, optional
            The name of the file to save the plot. If None, the plot is shown instead.
        """
        plt.figure(figsize=(15, 10))

        if plot_type == 'heatmap':
            sns.heatmap(self.dataframe.isnull(), cbar=False, cmap='viridis')
        elif plot_type == 'bar':
            missing_counts = self.dataframe.isnull().sum()
            ax = missing_counts.plot(kind='bar')
            for container in ax.containers:
                ax.bar_label(container)
        elif plot_type == 'matrix':
            msno.matrix(self.dataframe)
        else:
            raise ValueError("Unsupported plot type")

        plt.title(title)

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def impute_missing(self, strategy='mean', fill_value=None, dataframe=None, **kwargs):
        """
        Impute the missing values in the DataFrame using the specified strategy.

        Parameters:
        -----------
        strategy : str
            The imputation strategy to use ('mean', 'median', 'most_frequent', 'predictive', or 'custom').
        fill_value : any, optional
            The value to use for the 'custom' imputation strategy. Required if strategy is 'custom'.
        dataframe : pd.DataFrame, optional
            Specific DataFrame to impute missing values. If None, use the initialized DataFrame.
        kwargs : dict
            Additional arguments to pass to the IterativeImputer for 'predictive' strategy.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with missing values imputed.
        """
        if dataframe is None:
            dataframe = self.dataframe

        # Replace None with np.nan
        dataframe = dataframe.replace({None: np.nan})

        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'predictive':
            imputer = IterativeImputer(**kwargs)
        elif strategy == 'custom':
            if fill_value is None:
                raise ValueError("fill_value must be provided for custom imputation")
            imputer = SimpleImputer(strategy='constant', fill_value=fill_value)
        else:
            raise ValueError("Unsupported imputation strategy")

        imputed_data = imputer.fit_transform(dataframe)
        dataframe = pd.DataFrame(imputed_data, columns=dataframe.columns)

        return dataframe
