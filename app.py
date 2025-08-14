import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load & Clean Data ---
df = pd.read_csv("Heart_attack_patients.csv")

# Replace '?' with NaN and convert to numeric where possible
df.replace('?', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

# --- Detect Target Column Automatically ---
possible_target_names = ['target', 'output', 'HeartDisease', 'heart_disease', 'class']
target_column = None
for col in df.columns:
    if col.strip().lower() in [name.lower() for name in possible_target_names]:
        target_column = col
        break

# --- App Title & Overview ---
st.title("Heart Attack Prediction & Analysis")
st.write("This app analyzes heart attack patient data and displays visual insights.")

st.subheader("Data Overview")
st.dataframe(df.head())
st.write("Dataset Shape:", df.shape)
st.write("Missing Values after cleaning:")
st.write(df.isnull().sum())

st.subheader("Data Description")
st.write(df.describe())

st.subheader("Data Types")
st.write(df.dtypes)

# --- Correlation Matrix ---
st.subheader("Correlation Matrix")
numeric_df = df.select_dtypes(include=[np.number]).dropna()
correlation_matrix = numeric_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# --- Helper Function for Bar Charts ---
def plot_bar_count(title, series):
    st.subheader(title)
    st.bar_chart(series.dropna().value_counts())

# --- Age Distribution ---
st.subheader("Age Distribution Histogram")
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df['age'].dropna(), bins=20, color='blue', alpha=0.7)
ax.set_xlabel("Age")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Bar Chart of Age Distribution")
st.bar_chart(df['age'].dropna())

# --- Counts by Category ---
if 'sex' in df.columns: plot_bar_count("Patient Count by Sex", df['sex'])
if 'cp' in df.columns: plot_bar_count("Patient Count by Chest Pain Type", df['cp'])
if 'fbs' in df.columns: plot_bar_count("Patient Count by Fasting Blood Sugar", df['fbs'])
if 'restecg' in df.columns: plot_bar_count("Patient Count by Resting ECG Results", df['restecg'])
if 'thalach' in df.columns: plot_bar_count("Patient Count by Maximum Heart Rate Achieved", df['thalach'])
if 'exang' in df.columns: plot_bar_count("Patient Count by Exercise Induced Angina", df['exang'])
if 'oldpeak' in df.columns: plot_bar_count("Patient Count by ST Depression (oldpeak)", df['oldpeak'])
if 'slope' in df.columns: plot_bar_count("Patient Count by Slope of ST Segment", df['slope'])
if 'ca' in df.columns: plot_bar_count("Patient Count by Number of Major Vessels (ca)", df['ca'])
if 'thal' in df.columns: plot_bar_count("Patient Count by Thalassemia", df['thal'])

# --- Target Column Related Plots ---
if target_column:
    plot_bar_count(f"Patient Count by {target_column}", df[target_column])

    # Violin Plot
    st.subheader(f"Violin Plot: Age by {target_column}")
    fig, ax = plt.subplots()
    sns.violinplot(x=target_column, y='age', data=df, palette='muted', ax=ax)
    st.pyplot(fig)

    # Box Plot
    st.subheader(f"Box Plot: Age by {target_column}")
    fig, ax = plt.subplots()
    sns.boxplot(x=target_column, y='age', data=df, palette='muted', ax=ax)
    st.pyplot(fig)

    # Pair Plot
    st.subheader("Pair Plot (Numeric Columns)")
    pairplot_df = numeric_df.copy()
    pairplot_df[target_column] = df[target_column]
    sns.pairplot(pairplot_df.dropna(), hue=target_column, palette='coolwarm')
    st.pyplot(plt.gcf())

    # Pie Chart
    st.subheader(f"Pie Chart: {target_column} Distribution")
    target_counts = df[target_column].value_counts()
    fig, ax = plt.subplots()
    ax.pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
           startangle=140, colors=['#ff9999', '#66b3ff'])
    st.pyplot(fig)
else:
    st.warning("No target column found. Skipping target-related plots.")

# --- Scatter Plot: Age vs Thalach ---
if 'thalach' in df.columns:
    st.subheader("Scatter Plot: Age vs Maximum Heart Rate Achieved")
    fig, ax = plt.subplots()
    ax.scatter(df['age'], df['thalach'], alpha=0.5)
    ax.set_xlabel("Age")
    ax.set_ylabel("Maximum Heart Rate Achieved")
    st.pyplot(fig)

# --- Line Chart ---
if 'thalach' in df.columns:
    st.subheader("Average Maximum Heart Rate by Age")
    average_max_hr = df.groupby('age')['thalach'].mean().reset_index()
    fig, ax = plt.subplots()
    ax.plot(average_max_hr['age'], average_max_hr['thalach'], marker='o')
    ax.set_xlabel("Age")
    ax.set_ylabel("Average Maximum Heart Rate")
    st.pyplot(fig)

# --- PDF ---
st.subheader("PDF of Age")
fig, ax = plt.subplots()
sns.kdeplot(df['age'].dropna(), shade=True, color='blue', ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Density")
st.pyplot(fig)

# --- CDF ---
st.subheader("CDF of Age")
fig, ax = plt.subplots()
sns.ecdfplot(df['age'].dropna(), color='blue', ax=ax)
ax.set_xlabel("Age")
ax.set_ylabel("Cumulative Probability")
st.pyplot(fig)

# --- Cholesterol Histogram ---
if 'chol' in df.columns:
    st.subheader("Cholesterol Level Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['chol'].dropna(), bins=20, color='green', alpha=0.7)
    ax.set_xlabel("Cholesterol Level")
    ax.set_ylabel("Count")
    st.pyplot(fig)
    