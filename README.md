# DataRefine
# <img src="DataRefine/scripts/drlogo.jpeg" alt="DataRefine logo" width="200"/>

![PyPI](https://img.shields.io/pypi/v/DataRefine?color=#2e86c1&label=pypi&logo=pypi)
![License](https://img.shields.io/github/license/Shahanafarvin/DataRefine)
![Python Versions](https://img.shields.io/pypi/pyversions/DataRefine)

**DataRefine** is a Python package designed for data cleaning with interactive output and visualizations. It offers a streamlined interface to help users detect and handle missing values, outliers, perform normalization and transformation, and assess data quality. The package also integrates interactive visualizations to make it easy for users to understand their data, along with an interface for an enhanced user experience.

## Features

- **Interactive Data Upload**: Easy CSV file upload functionality
- **Missing Data Handling**:
  - Multiple imputation strategies (mean, median, mode, predictive)
  - Visual representation of missing value patterns
  - Column-specific imputation options
  
- **Outlier Detection & Treatment**:
  - Multiple detection methods (IQR, Z-score)
  - Configurable thresholds
  - Visual outlier analysis using box plots
  - Multiple handling strategies (capping, removal, imputation)

- **Data Normalization**:
  - Multiple normalization methods (Min-Max, Z-score, Robust scaling)
  - Interactive distribution visualization
  - Column-specific normalization

- **Data Transformation**:
  - Log transformation
  - Square root transformation
  - Box-Cox transformation
  - Before/after distribution comparison

- **Data Quality Assessment**:
  - Summary statistics
  - Visual quality reports

## Installation

It's recommended to install `DataRefine` in a virtual environment to manage dependencies effectively and avoid conflicts with other projects.

### 1. Set Up a Virtual Environment

**For Python 3.3 and above:**

1. **Create a Virtual Environment:**

    ```bash
    python -m venv env
    ```

    Replace `env` with your preferred name for the virtual environment.

2. **Activate the Virtual Environment:**

    - **On Windows:**
      ```bash
      env\Scripts\activate
      ```

    - **On macOS/Linux:**
      ```bash
      source env/bin/activate
      ```

### 2. Install DataRefine

Once the virtual environment is activated, you can install `DataRefine` using `pip`:

```bash
pip install DataRefine
```
## Quick Start

After installation, you can start DataRefine directly by running:

```bash
DataRefine
```
Open your web browser and navigate to the provided local URL.

Upload your CSV file.

Start cleaning your data!

## How to use?

- **Data Upload:**
    - Click the "Upload CSV" button.
    - Select your CSV file from your local system.

- **Data Cleaning:**
    - Use the sidebar to navigate between different cleaning operations.
    - Configure parameters using the interactive controls.
    - View real-time visualizations of the changes.
    - Download the cleaned dataset when finished.
    - For a detailed video walkthrough of the app's features and functionality, check out our YouTube demo.

[![Video Title](https://img.youtube.com/vi/FOnMv-vnURI/0.jpg)](https://youtu.be/FOnMv-vnURI?si=XfqJ9IhgLfOjJHh4)


## Requirements

- Python >= 3.7
- Streamlit
- Pandas
- NumPy
- plotly
- scikit-learn

For more detailed information, see the `requirements.txt` file.

## Contributing

We welcome contributions! Please follow these steps:

- Fork the repository
- Create a new branch (git checkout -b feature/improvement)
- Make your changes
- Commit your changes (git commit -am 'Add new feature')
- Push to the branch (git push origin feature/improvement)
- Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md)file for details.

## Acknowledgments

Special thanks to all the libraries and frameworks that have helped in developing this package.

## Version History

- 1.0.0: Initial release
- Basic data cleaning functionality
- Interactive web interface
- Visualization capabilities





