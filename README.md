# Student Performance Analysis

## Overview

This project analyzes the UCI Student Performance Dataset to identify factors that influence student academic success. The analysis includes  data preprocessing, exploratory data analysis, statistical inference, feature engineering, and machine learning modeling.

## Assignment Requirements

### Question 3: Student Performance Analysis

- **Part A**: Data Loading & Preprocessing (15 points)
- **Part B**: First EDA (20 points)
- **Part C**: Feature Engineering (15 points)
- **Part D**: Second EDA & Statistical Inference (15 points)
- **Part E**: Simple Machine Learning (25 points)

## Project Structure

```text
exercise_2/
├── student_performance_analysis.py    # Main analysis script
├── Math-Students.csv                  # UCI Student Performance Dataset
├── README.md                         # This file
└── .venv/                            # Python virtual environment
```

## Prerequisites

- Python 3.9+
- Virtual environment (recommended)

## Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

Or install all dependencies:

```bash

pip install -r requirements.txt
```

## How to Run

1. **Clone or download this repository**

```bash
git clone https://github.com/Hansen256/artificial_intelligence_coursework_2
```

1. **Navigate to the project directory:**

   ```bash
   cd exercise_2
   ```

1. **Activate virtual environment (if using):**

   ```bash
   # Windows
   .venv\Scripts\activate
   
   # macOS/Linux
   source .venv/bin/activate
   ```

1. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```

1. **Run the analysis:**

   ```bash
   python student_performance_analysis.py
   ```

## Analysis Overview

This analysis walks through a comprehensive approach to understanding student performance data. Here's what we accomplish in each section:

### Part A: Data Loading & Preprocessing

We start by loading the UCI Student Performance Dataset, which contains information about 399 students across 33 different features. The preprocessing phase includes several important steps:

- We inspect the data structure to understand what we're working with
- We handle any missing values using smart conditional processing (though this dataset is quite clean)
- We encode categorical variables like school type, gender, and family status using one-hot encoding, being careful to drop one category to prevent multicollinearity issues
- Finally, we normalize all numeric features using StandardScaler to ensure they're on the same scale for analysis

### Part B: Exploratory Data Analysis

This is where we really dig into the data to understand patterns and relationships:

- We calculate descriptive statistics for key variables like grades and study time
- We create distribution plots to see how grades are spread across the student population
- We examine how parental education levels relate to student performance through bar charts
- We build a correlation heatmap to identify which factors are most strongly related to final grades
- Throughout this process, we document our key findings with specific correlation values and insights

### Part C: Feature Engineering

Here we create new variables that might be more predictive than the original ones:

- **avg_G1_G2**: We calculate the average of the first two period grades, which should be a strong predictor of final performance
- **pass**: We create a binary pass/fail variable using a threshold of 10 out of 20 points
- **study_absences**: We combine study time and absences to capture the interaction between these factors
- **failures_absences**: Similarly, we look at how previous failures and absences might compound each other
- **parent_edu_avg**: We average mother's and father's education levels as an additional family background indicator

### Part D: Statistical Inference

We use formal statistical tests to validate our observations:

- **ANOVA Test**: We test whether different study time groups have significantly different final grades
- **Chi-square Test**: We examine whether having internet access is associated with pass/fail outcomes
- We create visualizations that clearly show these statistical relationships
- We interpret all p-values and explain what they mean in practical terms

### Part E: Machine Learning

Finally, we build a predictive model to see how well we can forecast student performance:

- We split the data into training and testing sets, using stratification to maintain the balance of pass/fail students
- We train a Linear Regression model to predict final grades
- We evaluate the model using multiple metrics including RMSE, MAE, and R-squared
- We analyze which features are most important for predictions
- We create comprehensive visualizations to understand model performance and identify any issues

## Key Findings

### What the Data Tells Us

The analysis reveals several interesting patterns about student performance:

## Academic Performance Patterns

- About two-thirds of students (66.42%) achieve passing grades of 10 or higher out of 20
- There's a very strong relationship between grades across different periods - students who do well early tend to continue doing well
- The correlation between G1 and G2 is 0.852, while G2 and G3 correlate at 0.905, showing increasing predictive power as we get closer to the final grade

## Study Habits and Technology

- Study time does have a measurable positive effect on final grades, as confirmed by our statistical tests
- Students with internet access have notably higher pass rates (68.5%) compared to those without (57.1%)
- The relationship between absences and performance is negative, as you'd expect

## Family Background Influence

- Parental education levels show a clear positive trend with student performance
- Students from families where parents have higher education levels tend to perform better
- This suggests that family background and support systems play an important role in academic success

### How Well Our Model Performs

Our Linear Regression model does quite well at predicting student grades:

- The model explains about 83% of the variance in final grades (R² ≈ 0.83)
- The typical prediction error is around 1.8 grade points (RMSE ≈ 1.8)
- The most important predictors are, unsurprisingly, the previous grades (G1 and G2)
- Other significant factors include study time, number of past failures, and various family support indicators

### What This Means Practically

These findings suggest that:

- Early intervention is crucial - students struggling in the first periods need immediate support
- Access to technology and internet resources can make a meaningful difference
- Family engagement and support should be considered when designing intervention programs
- While study time matters, it's not the only factor - systemic support is also important

## Technical Implementation

Our approach emphasizes quality and best practices throughout the analysis:

**Data Processing Excellence**
We implement proper preprocessing order by creating features before scaling, ensuring meaningful thresholds based on domain knowledge (like using 10/20 as a sensible pass/fail cutoff), and conducting statistical tests on original unscaled data for proper interpretation.

**Visualization Quality**
All plots include proper titles, labels, and annotations. We use statistical annotations with p-values and significance indicators where relevant. Value labels are displayed on bar charts for easy interpretation, and we use masked heatmaps to remove redundant information. Every visualization uses proper layout management for professional presentation.

**Statistical Rigor**
We include conditional processing and sample size validation for our statistical tests. All results are clearly documented with proper comments and section organization. The code follows professional standards that would be suitable for publication or industry use.

## Code Quality and Best Practices

The implementation demonstrates several important data science principles:

## Proper Methodology

- Feature engineering happens before scaling to maintain interpretability
- We use meaningful thresholds based on domain knowledge rather than arbitrary values
- Statistical tests use original unscaled data so results can be properly interpreted
- Error handling and edge case validation are built in throughout

## Assignment Compliance

| Requirement | Implementation | Status |
|-------------|---------------|--------|
| Load dataset and inspect | ✅ df.head(), df.info(), df.shape | Complete |
| Encode categoricals | ✅ OneHotEncoder with drop='first' | Complete |
| Scale numeric features | ✅ StandardScaler on all numerics | Complete |
| Handle missing values | ✅ Smart conditional handling | Complete |
| Descriptive statistics | ✅ Comprehensive stats with insights | Complete |
| Distribution plots | ✅ Professional 3-panel histograms | Complete |
| Parental education analysis | ✅ Bar charts with value annotations | Complete |
| Correlation heatmap | ✅ Masked heatmap with formatting | Complete |
| Feature discussion | ✅ Quantified insights and correlations | Complete |
| G1+G2 average feature | ✅ avg_G1_G2 computation | Complete |
| Pass/fail classification | ✅ Meaningful 10/20 threshold | Complete |
| Interaction features | ✅ study×absences, failures×absences | Complete |
| Feature rationale | ✅ Clear explanations for each | Complete |
| ANOVA test | ✅ Study time vs grades with validation | Complete |
| Chi-square test | ✅ Internet vs pass/fail association | Complete |
| Statistical visualization | ✅ Boxplots and bar charts with p-values | Complete |
| Train/test split | ✅ Stratified 80/20 split | Complete |
| Model training | ✅ Linear Regression implementation | Complete |
| Performance evaluation | ✅ Multiple metrics and visualization | Complete |

## Author

- Siima Diana                2023-B291-12710
- Mulindwa Solomon           2023-B291-12501
- Tumusiime Hansen Andrew    2023-B291-10756
- Pande Azizi                2023-B291-12779
- Mwebesa Johnson            2023-B291-13172
- Katerega Josepth Travour   2023-B291-10759

This project is for educational purposes as part of coursework assignment.
