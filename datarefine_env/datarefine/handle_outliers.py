import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from rich import box

class OutlierHandler:
    """
    A class used to detect and handle outliers in numeric columns of a DataFrame.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame containing data to be processed for outliers.
    """

    def __init__(self, dataframe):
        """
        Initialize the OutlierHandler with the provided DataFrame.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be processed for outliers.
        """
        self.dataframe = dataframe.select_dtypes(include=[np.number])  # Select only numeric columns
        self.console = Console()

    def visualize_outliers(self, plot_type='box', title="Outliers", filename=None):
        """
        Visualize the outliers in the DataFrame using the specified plot type.

        Parameters:
        ----------
        plot_type : str
            The type of plot to use ('box', 'hist', 'scatter').
        title : str
            The title of the plot.
        filename : str, optional
            The name of the file to save the plot. If None, the plot is shown instead.
        """
        plt.figure(figsize=(10, 6))

        if plot_type == 'box':
            sns.boxplot(data=self.dataframe)
        elif plot_type == 'hist':
            for column in self.dataframe.columns:
                sns.histplot(self.dataframe[column], kde=True)
                plt.title(f'{title}: {column}')
                if filename:
                    plt.savefig(f"{filename}_{column}.png")
                    plt.close()
                else:
                    plt.show()
        elif plot_type == 'scatter':
            num_columns = self.dataframe.columns
            if len(num_columns) < 2:
                raise ValueError("Scatter plot requires at least two numerical columns")
            sns.scatterplot(x=self.dataframe[num_columns[0]], y=self.dataframe[num_columns[1]])
        else:
            raise ValueError("Unsupported plot type")

        plt.title(title)
        plt.tight_layout()

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def detect_outliers(self, method='zscore', threshold=3.0, **kwargs):
        """
        Detect outliers in the numeric columns of the DataFrame using the specified method.

        Parameters:
        ----------
        method : str
            The method to use for outlier detection ('zscore', 'iqr', 'isolation_forest', 'lof').
        threshold : float
            The threshold for outlier detection.
        kwargs : dict
            Additional arguments for specific detection methods.

        Returns:
        -------
        pd.DataFrame
            A DataFrame indicating the outliers with True (outlier) or False (not an outlier).
        """
        if method == 'zscore':
            z_scores = np.abs((self.dataframe - self.dataframe.mean()) / self.dataframe.std())
            outliers = (z_scores > threshold)
        elif method == 'iqr':
            Q1 = self.dataframe.quantile(0.25)
            Q3 = self.dataframe.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.dataframe < (Q1 - threshold * IQR)) | (self.dataframe > (Q3 + threshold * IQR)))
        elif method == 'isolation_forest':
            iso_forest = IsolationForest(**kwargs)
            outlier_predictions = iso_forest.fit_predict(self.dataframe)
            outliers = pd.DataFrame(outlier_predictions, index=self.dataframe.index, columns=['outlier'])
            outliers['outlier'] = outliers['outlier'] == -1
        elif method == 'lof':
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(self.dataframe)
            if 'n_neighbors' not in kwargs:
                kwargs['n_neighbors'] = min(20, len(self.dataframe) - 1)
            lof = LocalOutlierFactor(**kwargs)
            lof_scores = lof.fit_predict(scaled_data)
            outliers = pd.DataFrame(lof_scores, index=self.dataframe.index, columns=['outlier'])
            outliers['outlier'] = outliers['outlier'] == -1
        else:
            raise ValueError("Unsupported outlier detection method")
        
        self.print_table(outliers, title=method.capitalize() + " Outliers")
        return outliers

    def handle_outliers(self, method='remove', detection_method='zscore', threshold=3.0, **kwargs):
        """
        Handle the outliers in the numeric columns of the DataFrame using the specified method.

        Parameters:
        ----------
        method : str
            The method to handle outliers ('remove', 'cap', 'impute').
        detection_method : str
            The method to use for outlier detection ('zscore', 'iqr', 'isolation_forest', 'lof').
        threshold : float
            The threshold for outlier detection.
        kwargs : dict
            Additional arguments for specific detection methods.

        Returns:
        -------
        pd.DataFrame
            The DataFrame with outliers handled.
        """
        # Detect outliers only in numeric columns
        outliers = self.detect_outliers(method=detection_method, threshold=threshold, **kwargs)
        
        # Handle outliers based on the specified method
        if method == 'remove':
            self.dataframe = self.dataframe[~outliers.any(axis=1)]
        elif method == 'cap':
            for column in self.dataframe.columns:
                Q1 = self.dataframe[column].quantile(0.25)
                Q3 = self.dataframe[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                self.dataframe[column] = np.where(self.dataframe[column] < lower_bound, lower_bound, self.dataframe[column])
                self.dataframe[column] = np.where(self.dataframe[column] > upper_bound, upper_bound, self.dataframe[column])
        elif method == 'impute':
            imputer = SimpleImputer(strategy='mean')
            self.dataframe = pd.DataFrame(imputer.fit_transform(self.dataframe), columns=self.dataframe.columns)
        else:
            raise ValueError("Unsupported outlier handling method")

        self.print_table(self.dataframe, title=f"DataFrame after handling outliers ({method.capitalize()} method)")
        return self.dataframe

    def print_table(self, df, title):
        """
        Print the DataFrame or outliers DataFrame using rich.

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
            table.add_column(col.capitalize(), justify="right", style="green" if df.equals(self.dataframe) else "red")

        for i, row in df.iterrows():
            table.add_row(str(i), *[f"{value:.2f}" for value in row])

        self.console.print(table)
