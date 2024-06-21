import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

class DataQualityAssessment:
    """
    Class for assessing the quality of a DataFrame, including summary statistics
    and quality metrics such as missing values and outliers.

    Attributes
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be assessed.

    Methods
    -------
    summary_statistics():
        Computes and prints summary statistics for the DataFrame.
    quality_metrics():
        Computes and prints quality metrics for the DataFrame.
    """

    def __init__(self, dataframe):
        """
        Initializes the DataQualityAssessment class with the given DataFrame.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be assessed.
        """
        self.dataframe = dataframe.select_dtypes(include=['number'])  # Select only numeric columns

    def summary_statistics(self):
        """
        Computes and prints summary statistics for the DataFrame, including skewness and kurtosis.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the summary statistics.
        """
        # Get basic statistics (mean, std, min, 25%, 50%, 75%, max)
        stats_df = self.dataframe.describe().T
        # Calculate skewness and add it to the statistics DataFrame
        stats_df['skewness'] = self.dataframe.skew()
        # Calculate kurtosis and add it to the statistics DataFrame
        stats_df['kurtosis'] = self.dataframe.kurtosis()

        # Print the summary statistics
        console = Console()
        table = Table(title="Summary Statistics", box=box.ROUNDED)
        table.add_column("Statistic", style="bold blue")  # Change heading color to blue
        table.add_column("Count", justify="right", style="green")  # Change value color to green
        table.add_column("Mean", justify="right", style="green")
        table.add_column("Std", justify="right", style="green")
        table.add_column("Min", justify="right", style="green")
        table.add_column("25%", justify="right", style="green")
        table.add_column("50%", justify="right", style="green")
        table.add_column("75%", justify="right", style="green")
        table.add_column("Max", justify="right", style="green")
        table.add_column("Skewness", justify="right", style="green")
        table.add_column("Kurtosis", justify="right", style="green")
        for index, row in stats_df.iterrows():
            table.add_row(index, *[f"{value:.2f}" if pd.notnull(value) else "NaN" for value in row])
        console.print(table)

        return stats_df

    def quality_metrics(self):
        """
        Computes and prints quality metrics for the DataFrame, such as the number of missing values
        and the number of outliers.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the quality metrics.
        """
        # Calculate the number of missing values for each column
        missing_values = self.dataframe.isnull().sum()
        # Calculate the number of outliers (values above the 99th percentile) for each column
        outliers = (self.dataframe > self.dataframe.quantile(0.99)).sum()
        # Combine the missing values and outliers into a DataFrame
        data_quality = pd.DataFrame({
            'missing_values': missing_values,
            'outliers': outliers
        })

        # Print the quality metrics
        console = Console()
        table = Table(title="Quality Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="bold blue")  # Change heading color to blue
        table.add_column("Missing Values", justify="right", style="green")  # Change value color to green
        table.add_column("Outliers", justify="right", style="green")
        for index, row in data_quality.iterrows():
            table.add_row(index, *[str(value) for value in row])
        console.print(table)

        return data_quality


