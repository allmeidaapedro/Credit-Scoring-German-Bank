'''
This script aims to provide functions that will turn the exploratory data analysis (EDA) process easier. 
'''


'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


def sns_plots(data, features, histplot=True, countplot=False,     
              barplot=False, barplot_y=None, boxplot=False, 
              boxplot_x=None, outliers=False, kde=False, 
              hue=None):
    '''
    Generate Seaborn plots for visualization.

    This function generates various types of Seaborn plots based on the provided
    data and features. Supported plot types include histograms, count plots,
    bar plots, box plots, and more.

    Args:
        data (DataFrame): The DataFrame containing the data to be visualized.
        features (list): A list of feature names to visualize.
        histplot (bool, optional): Generate histograms. Default is True.
        countplot (bool, optional): Generate count plots. Default is False.
        barplot (bool, optional): Generate bar plots. Default is False.
        barplot_y (str, optional): The name of the feature for the y-axis in bar plots.
        boxplot (bool, optional): Generate box plots. Default is False.
        boxplot_x (str, optional): The name of the feature for the x-axis in box plots.
        outliers (bool, optional): Show outliers in box plots. Default is False.
        kde (bool, optional): Plot Kernel Density Estimate in histograms. Default is False.
        hue (str, optional): The name of the feature to use for color grouping. Default is None.

    Returns:
        None

    Raises:
        CustomException: If an error occurs during the plot generation.

    '''
    
    try:
        # Getting num_features and num_rows and iterating over the sublot dimensions.
        num_features = len(features)
        num_rows = num_features // 3 + (num_features % 3 > 0)  

        fig, axes = plt.subplots(num_rows, 3, figsize=(20, 5*num_rows))  

        for i, feature in enumerate(features):
            row = i // 3  
            col = i % 3  

            ax = axes[row, col] if num_rows > 1 else axes[col] 
            
            if countplot:
                # Plotting countplot and adding the counts at the top of each bar.
                sns.countplot(data=data, x=feature, hue=hue, ax=ax)
                for container in ax.containers:
                    ax.bar_label(container)

            elif barplot:
                # Plotting barplot and adding the averages at the top of each bar.
                ax = sns.barplot(data=data, x=feature, y=barplot_y, hue=hue, ax=ax, ci=None)
                for container in ax.containers:
                    ax.bar_label(container)

            elif boxplot:
                # Plotting multivariate boxplot.
                sns.boxplot(data=data, x=boxplot_x, y=feature, showfliers=outliers, ax=ax)

            elif outliers:
                # Plotting univariate boxplot.
                sns.boxplot(data=data, x=feature, ax=ax)

            else:
                # Plotting histplot.
                sns.histplot(data=data, x=feature, hue=hue, kde=kde, ax=ax)

            ax.set_title(feature)  
            ax.set_xlabel('')  
        
        # Removing unused axes.
        if num_features < len(axes.flat):
            for j in range(num_features, len(axes.flat)):
                fig.delaxes(axes.flat[j])

        plt.tight_layout()
    
    except Exception as e:
        raise CustomException(e, sys)


def check_outliers(data, features):
    '''
    Check for outliers in the given dataset features.

    This function calculates and identifies outliers in the specified features
    using the Interquartile Range (IQR) method.

    Args:
        data (DataFrame): The DataFrame containing the data to check for outliers.
        features (list): A list of feature names to check for outliers.

    Returns:
        tuple: A tuple containing three elements:
            - outlier_indexes (dict): A dictionary mapping feature names to lists of outlier indexes.
            - outlier_counts (dict): A dictionary mapping feature names to the count of outliers.
            - total_outliers (int): The total count of outliers in the dataset.

    Raises:
        CustomException: If an error occurs while checking for outliers.

    '''
    
    try:
    
        outlier_counts = {}
        outlier_indexes = {}
        total_outliers = 0
        
        for feature in features:
            Q1 = data[feature].quantile(0.25)
            Q3 = data[feature].quantile(0.75)
            
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            feature_outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]
            outlier_indexes[feature] = feature_outliers.index.tolist()
            outlier_count = len(feature_outliers)
            outlier_counts[feature] = outlier_count
            total_outliers += outlier_count
        
        print(f'There are {total_outliers} outliers in the dataset.')
        print()
        print(f'Number (percentage) of outliers per feature: ')
        print()
        for feature, count in outlier_counts.items():
            print(f'{feature}: {count} ({round(count/len(data)*100, 2)})%')

        return outlier_indexes, outlier_counts, total_outliers
    
    except Exception as e:
        raise CustomException(e, sys)
    

def iv_analysis(data, cat, target, positive_label='Positive', cat_label='Categorical'):
    '''
    Calculate Information Value (IV) and Weight of Evidence (WOE) analysis for categorical data.

    :param data: DataFrame containing the data.
    :type data: pandas.DataFrame
    :param cat: Name of the categorical feature to analyze.
    :type cat: str
    :param target: Name of the target binary variable (1/0).
    :type target: str
    :param positive_label: Label for the positive category (default is 'Positive').
    :type positive_label: str, optional
    :param cat_label: Label for the categorical feature (default is 'Categorical').
    :type cat_label: str, optional

    :return: DataFrame with IV and WOE analysis results.
    :rtype: pandas.DataFrame

    This function calculates Information Value (IV) and Weight of Evidence (WOE) statistics for a categorical feature
    with respect to a binary target variable. It provides insights into the relationship between the categorical
    feature and the target variable.

    Example:
    ```python
    iv_result = iv_analysis(data, 'Category', 'Target')
    ```

    The resulting DataFrame contains columns for counts, percentages, WOE, IV, and total counts for each category,
    along with a summary row at the end.

    :raises CustomException: If an exception occurs during the analysis.
    '''
    try:
        # Getting a df with 1 and 0 counts in each category of cat.
        iv_analysis_df = data.groupby(cat)[target].value_counts().unstack(fill_value=0)

        # Getting the total individuals counts per cat category. Adding a new column with this value.
        total_counts = iv_analysis_df.sum(axis=1)
        iv_analysis_df['Total'] = total_counts

        # Getting the total number of 1 and 0. Adding a new column with 1 and 0 percentages per cat category with respect to the total 1 and 0 separately.
        total_positive = iv_analysis_df[1].sum()
        total_negative = iv_analysis_df[0].sum()
        iv_analysis_df['Yes (%)'] = iv_analysis_df[1] / total_positive
        iv_analysis_df['No (%)'] = iv_analysis_df[0] / total_negative

        # Adding a column with the positive probabilities or positive rates within each cat category.
        iv_analysis_df[f'{positive_label} Probability (%)'] = round(iv_analysis_df[1] / iv_analysis_df['Total'] * 100, 2)

        # Adding a column containing the information value (iv) and a column containing the weight of evidence (woe).
        yes_pct = iv_analysis_df['Yes (%)']
        no_pct = iv_analysis_df['No (%)']
        woe = np.log(yes_pct / no_pct)
        iv = (yes_pct - no_pct) * woe
        iv_analysis_df['WOE'] = round(woe, 2)
        iv_analysis_df['IV'] = round(iv, 2)
        iv_analysis_df['WOE'].replace(np.inf, 0, inplace=True)
        iv_analysis_df['IV'].replace(np.inf, 0, inplace=True)
        total_iv = iv_analysis_df['IV'].sum()

        # Adding a row with the totals per each column to the dataframe.
        total_instances = total_positive + total_negative
        total_row = pd.Series({
            1: round(total_positive),
            0: round(total_negative),
            'Total': round(total_instances),
            f'{positive_label} Probability (%)': round((total_positive / total_instances) * 100, 2),
            'IV': round(total_iv, 2)
        },  name='Total')

        iv_analysis_df = pd.concat([iv_analysis_df, total_row.to_frame().T])

        # Rounding to integer and percentages.
        iv_analysis_df[0] = iv_analysis_df[0].astype(int)
        iv_analysis_df[1] = iv_analysis_df[1].astype(int)
        iv_analysis_df['Total'] = iv_analysis_df['Total'].astype(int)
        iv_analysis_df['Yes (%)'] = round(100 * iv_analysis_df['Yes (%)'], 2)
        iv_analysis_df['No (%)'] = round(100 * iv_analysis_df['No (%)'], 2)
        iv_analysis_df.fillna('-', inplace=True)

        # Reset the index temporarily and add "cat_label" as a new row
        iv_analysis_df.reset_index(inplace=True)
        iv_analysis_df.rename(columns={'index': cat_label}, inplace=True)
        iv_analysis_df.set_index(cat_label, inplace=True)


        # Create a MultiIndex for the columns with 'positive_label' as the top-level label
        iv_analysis_df.columns = pd.MultiIndex.from_tuples([(positive_label, col) for col in iv_analysis_df.columns])

        # Concatenate the 'Default' DataFrame above the existing DataFrame
        iv_analysis_df = pd.concat([iv_analysis_df])

        return iv_analysis_df

    except Exception as e:
        raise CustomException(e, sys)