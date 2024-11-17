# Employee Retention Prediction

## Description
This project implements machine learning models to predict employee attrition and identify key factors contributing to turnover. Using the IBM HR Analytics dataset, it provides insights for developing effective retention strategies.

## Features
- Employee attrition prediction using multiple ML models
- Feature importance analysis
- Model comparison and evaluation
- SHAP value interpretation
- Visual representation of results

## Technologies Used
- Python 3.x
- Scikit-learn for machine learning models
- XGBoost for gradient boosting
- SHAP for model interpretation
- Pandas and NumPy for data manipulation
- Matplotlib and Seaborn for visualization

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Alissa-King/employee-retention-prediction.git
   cd employee-retention-prediction
   ```
2. Install required packages:
   ```
   pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
   ```

## Usage
1. Run the analysis script:
   ```
   python retention_analysis.py
   ```
2. View model performance metrics and visualizations
3. Access prediction results in 'predictions.csv'

## Model Components
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble learning model
- **XGBoost**: Gradient boosting model
- **SHAP Analysis**: Model interpretation

## Data Features
- Employee demographics
- Job role information
- Satisfaction metrics
- Performance indicators
- Work-life balance factors

## Results
- Model Accuracy: 87.76%
- Detailed feature importance analysis
- Comprehensive evaluation metrics
- Visual interpretation of predictions

## Contributing
Contributions to improve the analysis are welcome. Please submit a Pull Request.

## License
This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments
- Dataset provided by IBM Watson Analytics
- Built with scikit-learn and XGBoost
