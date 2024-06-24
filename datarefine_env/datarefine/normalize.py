import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, PowerTransformer
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich import box

class DataNormalizer:
    """
    A class used to normalize and transform data in a DataFrame.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing data to be normalized and transformed.
    """

    def __init__(self, dataframe):
        """
        Initialize the DataNormalizer with the provided DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be normalized and transformed.
        """
        self.dataframe = dataframe.select_dtypes(include=['number'])  # Select only numeric columns
        self.console = Console()

    def normalize(self, method='minmax'):
        """
        Normalize the DataFrame using the specified method.

        Parameters
        ----------
        method : str
            The normalization method to use ('minmax', 'zscore', 'robust').

        Returns
        -------
        pd.DataFrame
            The normalized DataFrame.
        """
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError("Unsupported normalization method")

        normalized_df = pd.DataFrame(scaler.fit_transform(self.dataframe), columns=self.dataframe.columns)
        self._print_dataframe(normalized_df, f"Normalized DataFrame ({method.capitalize()})")
        return normalized_df

    def transform(self, method='log'):
        """
        Transform the DataFrame using the specified method.

        Parameters
        ----------
        method : str
            The transformation method to use ('log', 'sqrt', 'boxcox').

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        transformed_df = self.dataframe.copy()
        if method == 'log':
            transformed_df = np.log1p(self.dataframe)
        elif method == 'sqrt':
            transformed_df = np.sqrt(self.dataframe)
        elif method == 'boxcox':
            pt = PowerTransformer(method='box-cox', standardize=False)
            transformed_df = pd.DataFrame(pt.fit_transform(self.dataframe), columns=self.dataframe.columns)
        else:
            raise ValueError("Unsupported transformation method")

        self._print_dataframe(transformed_df, f"Transformed DataFrame ({method.capitalize()})")
        return transformed_df

    def visualize_distribution(self, plot_type='hist', title="Distribution", filename=None):
        """
        Visualize the distribution of the DataFrame using specified plot type.

        Parameters
        ----------
        plot_type : str
            The type of plot to use ('hist', 'box', 'density').
        title : str
            The title of the plot.
        filename : str, optional
            The name of the file to save the plot. If None, the plot is shown instead.
        """
        plt.figure(figsize=(15, 10))

        if plot_type == 'hist':
            self.dataframe.hist(bins=30, layout=(1, len(self.dataframe.columns)), figsize=(15, 10), color='skyblue', edgecolor='black')
        elif plot_type == 'box':
            self.dataframe.plot(kind='box', subplots=True, layout=(1, len(self.dataframe.columns)), figsize=(15, 10), color='orange', patch_artist=True)
        elif plot_type == 'density':
            self.dataframe.plot(kind='density', subplots=True, layout=(1, len(self.dataframe.columns)), figsize=(15, 10), linewidth=2)
        else:
            raise ValueError("Unsupported plot type")

        plt.suptitle(title, fontsize=16, color='blue')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def _print_dataframe(self, df, title):
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
            table.add_column(col, justify="right", style="green")

        for i, row in df.iterrows():
            table.add_row(str(i), *[f"{value:.2f}" for value in row])

        self.console.print(table)

