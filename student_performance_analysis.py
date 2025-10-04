import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f_oneway, chi2_contingency

# Part A: Data Loading & Preprocessing
df = pd.read_csv('Math-Students.csv')
print("Dataset shape:", df.shape)

# Create features before scaling
df['avg_G1_G2'] = (df['G1'] + df['G2']) / 2
df['pass'] = (df['G3'] >= 10).astype(int)
df['study_absences'] = df['studytime'] * df['absences']
df['failures_absences'] = df['failures'] * df['absences']

# Encode categoricals
categoricals = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian',
                'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded = encoder.fit_transform(df[categoricals])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categoricals))
df = df.drop(categoricals, axis=1)
df = pd.concat([df, encoded_df], axis=1)

# Store original for tests
df_original = df.copy()

# Scale numeric features
numerics = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime',
            'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2', 'G3', 'avg_G1_G2', 
            'study_absences', 'failures_absences']
scaler = StandardScaler()
df[numerics] = scaler.fit_transform(df[numerics])

# Part B: First EDA
print("\nDescriptive Statistics:")
print(df[['G1', 'G2', 'G3', 'studytime']].describe())

# Distribution plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, grade in enumerate(['G1', 'G2', 'G3']):
    sns.histplot(df[grade], kde=True, ax=axes[i])
    axes[i].set_title(f'{grade} Distribution')
plt.tight_layout()
plt.show()

# Parental education vs grades
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
df.groupby('Medu')['G3'].mean().plot(kind='bar', ax=axes[0], title="Mother's Education vs G3")
df.groupby('Fedu')['G3'].mean().plot(kind='bar', ax=axes[1], title="Father's Education vs G3")
plt.tight_layout()
plt.show()

# Correlation heatmap
corr = df[numerics].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

print("\nKey correlations with G3:")
print(f"G1-G3: {corr.loc['G1', 'G3']:.3f}")
print(f"G2-G3: {corr.loc['G2', 'G3']:.3f}")
print(f"Study time-G3: {corr.loc['studytime', 'G3']:.3f}")

# Part C: Feature Engineering (already done above)
print("\nFeatures created: avg_G1_G2, pass, study_absences, failures_absences")

# Part D: Statistical Tests
# ANOVA
study_groups = [group['G3'].values for name, group in df_original.groupby('studytime')]
f_stat, p_anova = f_oneway(*study_groups)
print(f"\nANOVA - Study time effect: F={f_stat:.3f}, p={p_anova:.3f}")

# Chi-square
contingency = pd.crosstab(df_original['internet'], df_original['pass'])
chi2, p_chi2, _, _ = chi2_contingency(contingency)
print(f"Chi-square - Internet vs Pass: chi2={chi2:.3f}, p={p_chi2:.3f}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.boxplot(data=df_original, x='studytime', y='G3', ax=axes[0])
axes[0].set_title(f'Study Time vs G3 (p={p_anova:.3f})')
pass_rate = df_original.groupby('internet')['pass'].mean()
pass_rate.plot(kind='bar', ax=axes[1], title=f'Internet vs Pass Rate (p={p_chi2:.3f})')
plt.tight_layout()
plt.show()

# Part E: Machine Learning
X = df.drop(['G3', 'pass'], axis=1)
y = df_original['G3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

# Feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)
print("\nTop 5 Important Features:")
print(importance.head(5))

# Results visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(y_test, y_pred, alpha=0.6)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[0].set_xlabel('Actual')
axes[0].set_ylabel('Predicted')
axes[0].set_title(f'Actual vs Predicted (R²={r2:.3f})')

importance.head(10)['coefficient'].plot(kind='barh', ax=axes[1])
axes[1].set_title('Feature Importance')
plt.tight_layout()
plt.show()

print(f"\nAnalysis complete. Model explains {r2:.1%} of grade variance.")
