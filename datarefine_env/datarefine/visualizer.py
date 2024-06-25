import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class DataVisualizer:
    """
    A class to perform various data visualizations for Exploratory Data Analysis (EDA).
    """

    def __init__(self, df):
        """
        Initializes the DataVisualizer with a pandas DataFrame.

        :param df: pandas DataFrame containing the data to visualize
        """
        self.df = df

    def plot_histogram(self, column, bins=10, title=None, color=None, hue=None, filename=None):
        """
        Plots a histogram for a specified column.

        :param column: The column to plot the histogram for
        :param bins: The number of bins for the histogram (default: 10)
        :param title: The title of the plot (default: None)
        :param color: The color palette for the plot (default: None)
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.histplot(self.df, x=column, bins=bins, kde=True, hue=hue, palette=color)
        plt.title(title if title else f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', 
                        xytext=(0, 10), textcoords='offset points')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


    def plot_boxplot(self, y_column, x_column=None, hue=None, title=None, palette=None, filename=None):
        """
        Plots a boxplot for a specified column, optionally grouped by another column.

        :param y_column: The column to plot the boxplot for
        :param x_column: The column to group the boxplot by (default: None)
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.boxplot(data=self.df, x=x_column, y=y_column, hue=hue, palette=palette)
        plt.title(title if title else f'Boxplot of {y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add data labels for medians
        medians = self.df.groupby([x_column])[y_column].median()
        vertical_offset = self.df[y_column].median() * 0.05  # Offset from median for annotation
        for xtick in ax.get_xticks():
            ax.text(xtick, medians[xtick] + vertical_offset, f'{medians[xtick]:.2f}', 
                    horizontalalignment='center', size='small', color='b', weight='semibold')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_scatter(self, x_column, y_column, hue=None, title=None, palette=None, filename=None):
        """
        Plots a scatter plot for two specified columns.

        :param x_column: The column for the x-axis
        :param y_column: The column for the y-axis
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.scatterplot(data=self.df, x=x_column, y=y_column, hue=hue, palette=palette)
        plt.title(title if title else f'Scatter Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

       
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_pairplot(self, hue=None, title=None, palette=None, filename=None):
        """
        Plots a pairplot for all numerical columns in the dataframe.

        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        pairplot = sns.pairplot(self.df, hue=hue, palette=palette)
        if title:
            pairplot.fig.suptitle(title, y=1.02)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    

    def plot_barplot(self, x_column, y_column=None, hue=None, title=None, palette=None, filename=None):
        """
        Plots a bar plot for a specified column, optionally grouped by another column.

        :param x_column: The column for the x-axis
        :param y_column: The column for the y-axis (default: None)
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        
        if y_column:
            ax = sns.barplot(data=self.df, x=x_column, y=y_column, hue=hue, palette=palette,errorbar=None)
            plt.title(title if title else f'Bar Plot of {x_column} vs {y_column}')
            plt.ylabel(y_column)
        else:
            ax = sns.countplot(data=self.df, x=x_column, hue=hue, palette=palette,errorbar=None)
            plt.title(title if title else f'Bar Plot of {x_column}')
            plt.ylabel('Count')
        
        plt.xlabel(x_column)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', 
                        xytext=(0, 10), textcoords='offset points')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_countplot(self, column, hue=None, title=None, palette=None, filename=None):
        """
        Plots a count plot for a specified column.

        :param column: The column to plot the count plot for
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.countplot(data=self.df, x=column, hue=hue, palette=palette)
        plt.title(title if title else f'Count Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add data labels
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.0f}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', 
                        xytext=(0, 10), textcoords='offset points')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_violinplot(self, y_column, x_column=None, hue=None, title=None, palette=None, filename=None):
        """
        Plots a violin plot for a specified column, optionally grouped by another column.

        :param y_column: The column to plot the violin plot for
        :param x_column: The column to group the violin plot by (default: None)
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.violinplot(data=self.df, x=x_column, y=y_column, hue=hue, palette=palette, split=True)
        plt.title(title if title else f'Violin Plot of {y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Add data labels for medians
        medians = self.df.groupby([x_column])[y_column].median()
        vertical_offset = self.df[y_column].median() * 0.05  # Offset from median for annotation
        for xtick in ax.get_xticks():
            ax.text(xtick, medians[xtick] + vertical_offset, f'{medians[xtick]:.2f}', 
            horizontalalignment='center', size='small', color='b', weight='semibold')

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_lineplot(self, x_column, y_column, hue=None, title=None, palette=None, filename=None):
        """
        Plots a line plot for two specified columns.

        :param x_column: The column for the x-axis
        :param y_column: The column for the y-axis
        :param hue: Variable in the dataframe to map plot aspects to different colors (default: None)
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        ax = sns.lineplot(data=self.df, x=x_column, y=y_column, hue=hue, palette=palette)
        plt.title(title if title else f'Line Plot of {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        
        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_piechart(self, column, title=None, palette=None, filename=None):
        """
        Plots a pie chart for a specified column.

        :param column: The column to plot the pie chart for
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        self.df[column].value_counts().plot.pie(autopct='%1.1f%%', colors=palette)
        plt.title(title if title else f'Pie Chart of {column}')
        plt.ylabel('')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_donutchart(self, column, title=None, palette=None, filename=None):
        """
        Plots a donut chart for a specified column.

        :param column: The column to plot the donut chart for
        :param title: The title of the plot (default: None)
        :param palette: The color palette for the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        data = self.df[column].value_counts()
        plt.pie(data, labels=data.index, autopct='%1.1f%%', wedgeprops=dict(width=0.3), colors=palette)
        plt.title(title if title else f'Donut Chart of {column}')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_matrix(self, title=None, filename=None):
        """
        Plots a matrix plot for the entire dataframe.

        :param title: The title of the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df, annot=True, fmt='g', cmap='viridis')
        plt.title(title if title else 'Matrix Plot of DataFrame')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()

    def plot_heatmap(self, title=None, filename=None):
        """
        Plots a heatmap of the correlation matrix for numerical columns only.

        :param title: The title of the plot (default: None)
        :param filename: The filename to save the plot (default: None)
        """
        # Select only numerical columns
        numerical_df = self.df.select_dtypes(include=['number'])
        
        # Calculate the correlation matrix for numerical columns
        corr = numerical_df.corr()

        
        ax = sns.heatmap(corr, annot=False, cmap='coolwarm')

        # Manually add data labels for each cell
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                plt.text(j + 0.5, i + 0.5, f'{corr.iloc[i, j]:.2f}',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='black')

        plt.title(title if title else 'Heatmap of Correlation Matrix')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()


    def plot_density(self, column, title=None, color=None, filename=None):
        """
        Plots a density plot for the specified column.

        A density plot shows the distribution of a variable's values.

        :param column: Column name to plot the density plot for
        :param title: Title of the plot, optional
        :param color: Color of the plot, optional
        :param filename: File path to save the plot as an image, optional
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(self.df[column],fill=True, color=color)
        plt.title(title if title else f'Density Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Density  ')
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if filename:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
