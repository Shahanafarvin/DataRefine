# datarefine/handle_missing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class MissingDataHandler:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def visualize_missing(self, filename):
        plt.figure(figsize=(10, 6))
        missing_counts = self.dataframe.isnull().sum()
        sns.barplot(x=missing_counts.index, y=missing_counts.values)

        # Annotate each bar with the missing count
        for i, count in enumerate(missing_counts.values):
            plt.text(i, count, f'{count}', ha='center', va='bottom')

        plt.title("Missing Values Count per Column")
        plt.xlabel("Columns")
        plt.ylabel("Count of Missing Values")
        plt.xticks(rotation=45)
        plt.savefig(filename)
        plt.close()

    def impute(self, strategy='mean', **kwargs):
        if strategy in ['mean', 'median', 'most_frequent']:
            imputer = SimpleImputer(strategy=strategy)
        elif strategy == 'predictive':
            imputer = IterativeImputer(**kwargs)
        else:
            raise ValueError("Unsupported imputation strategy")

        # Visualize missing data before imputation
        self.visualize_missing("missing_before_imputation.png")

        imputed_data = imputer.fit_transform(self.dataframe)
        self.dataframe = pd.DataFrame(imputed_data, columns=self.dataframe.columns)
        
        # Visualize missing data after imputation
        self.visualize_missing("missing_after_imputation.png")
        
        return self.dataframe
    
  
