# DataRefine-Capstone-Project
DataRefine is a Python package designed to streamline the data cleaning process by offering a wide range of functionalities to address common data quality issues. With integrated visualization capabilities, DataRefine aims to enhance transparency and empower data scientists to make informed decisions at each step of the cleaning process.

## Changelog

### version 1:(05/06/2024)
#### __init__.py
- This file serves to signify a directory as a Python package and can contain initialization code, module imports, and package-level variables, providing configuration and setup for the package upon import.
#### handle_missing.py
- This file contains classes and functions for handling missing data in a DataFrame, including methods for visualizing missing values and imputing missing data using various strategies like mean, median, mode, and predictive

### version 2:(07/06/2024)
#### handle_missing.py
- New strategy called custom added to fill missing values with custom defined values.
- Changed the visual download as optional ( if filename exists only).

### version 3:(10/06/2024)
#### __init__.py
- Updated to include other modules like handle_outlier.p and normalize.py
#### handle_missing.py
- Added more plot option for visualising missing values.
#### handle_outliers.py
- This file comprises classes and functions for detecting and handling outliers in a dataset, utilizing methods such as Z-score, IQR, and Isolation Forest for detection, and offering options for removal, transformation, or imputation of outliers based on user preference. visualizations are included to illustrate the distribution of data before and after handling outliers.(included more plot options)
#### normalize.py
- This file includes functions for both normalizing numerical data, utilizing methods like Min-Max scaling and StandardScaler, and transforming data through techniques such as log transformation and box-cox transformation, ensuring comprehensive preprocessing capabilities for numerical features in a DataFrame. These utilities empower users to standardize numerical values across features and improve the distributional properties of their data effectively.visualisation also included with more plot option.

### version 4:(11/06/2024)
#### handle_missing.py
- Updated to handle both text and numeric columns separately.
#### handle_outliers.py
- Updated the code to remove some errors.

### version 5:(11/06/2024)
#### __init__.py
- Updated the code to include another modules called data_quality_assessment.py
#### data_quality_assessment.py
- This file primarily includes functions for generating summary statistics and quality metrics for a DataFrame, accompanied by visualizations like histograms and density plots to aid in understanding the data's distribution and quality characteristics effectively.

### version 6:(21/06/2024)
#### data_quality_assessment.py
- this module updated to avoid visualisations,added rich library feature to use in layout of terminal output

### version 7:(24/06/2024)
- All other modules are updated to get same feature of rich library in data_quality_assessment.py

### version 8:(24/06/24)
#### handle_outliers.py
-this module is changed to handle numerical columns only

### version 9:(25/06/24)
#### visualizer.py
- This DataVisualizer module is designed for comprehensive exploratory data analysis (EDA) through a variety of visualizations. It includes methods for plotting histograms, boxplots, scatter plots, pairplots, heatmaps, bar plots, count plots, violin plots, line plots, pie charts, donut charts, density plots, and matrix plots, leveraging the matplotlib and seaborn libraries to facilitate insightful data exploration and presentation





