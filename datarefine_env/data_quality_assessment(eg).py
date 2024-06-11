import pandas as pd
from datarefine.data_quality_assessment import DataQualityAssessment

def main():
    # Sample DataFrame
    data = {
        'A': [1, 2, None, 4, 5],
        'B': [5, None, 7, 8, 10],
        'C': [1, 2, 2, 2, 5],
        'D': ['apple', 'banana', 'apple', 'cherry', 'banana']
    }
    df = pd.DataFrame(data)

    # Initialize DataQualityAssessment
    quality_assessor = DataQualityAssessment(df)

    # Get Summary Statistics
    summary_stats = quality_assessor.summary_statistics()
    print("Summary Statistics:\n", summary_stats)

    # Compute Quality Metrics
    quality_metrics = quality_assessor.quality_metrics()
    print("Quality Metrics:\n", quality_metrics)

    # Visualize Data Quality
    quality_assessor.visualize_quality(plot_type='histogram',filename="quality_histogram.png")
    quality_assessor.visualize_quality(plot_type='density',filename="quality_density.png")
    quality_assessor.visualize_quality(plot_type='qqplot',filename="quality_qq.png")

if __name__ == "__main__":
    main()
