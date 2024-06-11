import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

class DataQualityAssessment:
    """
    A class used to assess data quality in a DataFrame.

    Attributes:
    ----------
    dataframe : pd.DataFrame
        The DataFrame to be assessed for data quality.
    """
    
    def __init__(self, dataframe):
        """
        Initialize the DataQualityAssessment with the provided DataFrame.

        Parameters:
        ----------
        dataframe : pd.DataFrame
            The DataFrame to be assessed for data quality.
        """
        self.dataframe = dataframe

    def summary_statistics(self):
        """
        Calculate summary statistics for each numeric variable in the DataFrame.

        Returns:
        -------
        pd.DataFrame
            A DataFrame containing summary statistics.
        """
        numeric_df = self.dataframe.select_dtypes(include=[np.number])
        stats_df = numeric_df.describe().transpose()
        stats_df['skewness'] = numeric_df.skew()
        stats_df['kurtosis'] = numeric_df.kurt()
        return stats_df

    def quality_metrics(self):
        """
        Compute quality metrics for the DataFrame.

        Returns:
        -------
        dict
            A dictionary containing quality metrics.
        """
        metrics = {
            'missing_percentage': self.dataframe.isnull().mean() * 100,
            'num_outliers': self._count_outliers(),
            'data_distribution': self.summary_statistics()
        }
        return metrics

    def _count_outliers(self):
        """
        Count the number of outliers in each numeric column using the IQR method.

        Returns:
        -------
        pd.Series
            A Series containing the number of outliers for each column.
        """
        numeric_df = self.dataframe.select_dtypes(include=[np.number])
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        is_outlier = ((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR)))
        return is_outlier.sum()

    def visualize_quality(self, plot_type='histogram', filename=None):
        """
        Generate visualizations for data quality assessment.

        Parameters:
        ----------
        plot_type : str
            The type of plot to use ('histogram', 'density', 'qqplot').
        filename : str, optional
            The name of the file to save the plot. If None, the plot is shown instead.
        """
        num_columns = self.dataframe.select_dtypes(include=[np.number]).columns
        for column in num_columns:
            plt.figure(figsize=(10, 6))

            if plot_type == 'histogram':
                sns.histplot(self.dataframe[column], kde=True)
                plt.title(f'Histogram of {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
                for p in plt.gca().patches:
                    plt.gca().annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                                       ha='center', va='center', fontsize=10, color='black', xytext=(0, 10),
                                       textcoords='offset points')
            elif plot_type == 'density':
                sns.kdeplot(self.dataframe[column], fill=True)
                plt.title(f'Density Plot of {column}')
                plt.xlabel(column)
                plt.ylabel('Density')
            elif plot_type == 'qqplot':
                stats.probplot(self.dataframe[column], dist="norm", plot=plt)
                plt.title(f'QQ Plot of {column}')
            else:
                raise ValueError("Unsupported plot type")

            plt.tight_layout()

            if filename:
                plt.savefig(f'{filename}_{column}.png')
                plt.close()
            else:
                plt.show()
