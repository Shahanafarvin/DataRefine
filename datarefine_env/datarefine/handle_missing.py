import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import missingno as msno
import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

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
        self.console = Console()

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

        self._print_table(dataframe, title=f"DataFrame after imputation ({strategy.capitalize()} strategy)")
        return dataframe

    def _print_table(self, df, title):
        """
        Print the DataFrame using rich.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to be printed.
        title : str
            The title of the table.
        """
        table = Table(title=title, box=box.ROUNDED, title_style="bold blue")
        table.add_column("Index", style="bold blue")
        
        for col in df.columns:
            # Determine the appropriate style based on the column type
            if pd.api.types.is_numeric_dtype(df[col]):
                table.add_column(col, justify="right", style="green")
            else:
                table.add_column(col, justify="right", style="blue")
        
        for i, row in df.iterrows():
            values = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    values.append(f"{row[col]:.2f}")
                else:
                    values.append(f"{row[col]}")
            table.add_row(str(i), *values)

        self.console.print(table)

