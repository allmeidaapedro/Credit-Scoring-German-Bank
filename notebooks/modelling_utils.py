'''
This script aims to provide functions that will turn the modelling process easier
'''

'''
Importing libraries
'''

# Data manipulation and visualization.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve
import time

# Debugging.
from src.exception import CustomException
import sys

# Warnings.
from warnings import filterwarnings
filterwarnings('ignore')


def evaluate_models_cv(models, X_train, y_train):
    '''
    Evaluate multiple machine learning models using stratified k-fold cross-validation (the stratified k-fold is useful for dealing with target imbalancement).

    This function evaluates a dictionary of machine learning models by training each model on the provided training data
    and evaluating their performance using stratified k-fold cross-validation. The evaluation metric used is ROC-AUC score.

    Args:
        models (dict): A dictionary where the keys are model names and the values are instantiated machine learning model objects.
        X_train (array-like): The training feature data.
        y_train (array-like): The corresponding target labels for the training data.

    Returns:
        pd.DataFrame: A DataFrame containing the evaluation results for each model, including their average validation scores
                  and training scores.

    Raises:
        CustomException: If an error occurs while evaluating the models.

    '''


    try:
        # Stratified KFold in order to maintain the target proportion on each validation fold - dealing with imbalanced target.
        n_folds = 5
        stratified_kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

        # Dictionaries with validation and training scores of each model for plotting further.
        models_val_scores = dict()
        models_train_scores = dict()

        for model in models:
            # Getting the model object from the key with his name.
            model_instance = models[model]

            # Measuring training time.
            start_time = time.time()
            
            # Fitting the model to the training data.
            model_instance.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time

            # Make predictions on training data and evaluate them.
            y_train_pred = model_instance.predict(X_train)
            train_score = roc_auc_score(y_train, y_train_pred)

            # Evaluate the model using k-fold cross validation, obtaining a robust measurement of its performance on unseen data.
            val_scores = cross_val_score(model_instance, X_train, y_train, scoring='roc_auc', cv=stratified_kfold)
            avg_val_score = val_scores.mean()
            val_score_std = val_scores.std()

            # Adding the model scores to the validation and training scores dictionaries.
            models_val_scores[model] = avg_val_score
            models_train_scores[model] = train_score

            # Printing the results.
            print(f'{model} results: ')
            print('-'*50)
            print(f'Training score: {train_score}')
            print(f'Average validation score: {avg_val_score}')
            print(f'Standard deviation: {val_score_std}')
            print(f'Training time: {round(training_time, 5)} seconds')
            print()

        # Plotting the results.
        print('Plotting the results: ')

        # Converting scores to a dataframe
        val_df = pd.DataFrame(list(models_val_scores.items()), columns=['Model', 'Average Val Score'])
        train_df = pd.DataFrame(list(models_train_scores.items()), columns=['Model', 'Train Score'])
        eval_df = val_df.merge(train_df, on='Model')

        # Plotting each model and their train and validation (average) scores.
        plt.figure(figsize=(15, 6))
        width = 0.35

        x = np.arange(len(eval_df['Model']))

        val_bars = plt.bar(x - width/2, eval_df['Average Val Score'], width, label='Average Validation Score', color='skyblue')
        train_bars = plt.bar(x + width/2, eval_df['Train Score'], width, label='Train Score', color='orange')

        plt.xlabel('Model')
        plt.ylabel('ROC-AUC Score')
        plt.title('Models Performances')
        plt.xticks(x, eval_df['Model'], rotation=45)

        # Add scores on top of each bar
        for bar in val_bars + train_bars:
            height = bar.get_height()
            plt.annotate('{}'.format(round(height, 2)),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.legend()
        plt.tight_layout()
        plt.show()

        return eval_df
    
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_classifier(y_true, y_pred, probas):
    '''
    Evaluate the performance of a binary classifier and visualize results.

    This function calculates and displays various evaluation metrics for a binary classifier,
    including the classification report, confusion matrix, and ROC AUC curve.

    Args:
    - y_true: True binary labels.
    - y_pred: Predicted binary labels.
    - probas: Predicted probabilities of positive class.

    Returns:
    - None (displays evaluation metrics).

    Raises:
    - CustomException: If an error occurs during evaluation.
    '''

    try:
        # Classification report
        print(classification_report(y_true, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot = True, fmt = 'd')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Values')
        plt.ylabel('Real Values')
        plt.show()
        
        # ROC AUC Curve and score
        fpr, tpr, thresholds = roc_curve(y_true, probas)
        auc = roc_auc_score(y_true, probas)

        plt.figure(figsize=(5, 3))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random guessing line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    except Exception as e:
        raise CustomException(e, sys)
    

def target_probability_distributions(predicted_probas, y_true, positive_label='1', negative_label='0'):
    '''
    Plot target probability distributions for binary classification.

    This function generates a KDE (Kernel Density Estimation) plot to visualize the probability distributions
    of predicted probabilities for positive and negative instances in a binary classification problem.

    Parameters:
        predicted_probas (numpy.ndarray): Predicted positive probabilities from a binary classification model.
        y_true (numpy.ndarray): True class labels (0 for negative, 1 for positive).
        positive_label (str, optional): Label for the positive class. Default is '1'.
        negative_label (str, optional): Label for the negative class. Default is '0'.

    Returns:
        None

    Example:
        target_probability_distributions(predicted_probs, true_labels, positive_label='Default', negative_label='Non-Default')

    Raises:
        CustomException: If an exception occurs during execution, it is raised with the error message.
    '''

    try:
        # Obtaining predicted probabilities of being positive for positive and negative instances.
        probas_positive = predicted_probas[y_true == 1]
        probas_negative = predicted_probas[y_true == 0]

        # Plotting kde plot with shaded curves
        sns.kdeplot(probas_positive, label=positive_label, color='green', shade=True)
        sns.kdeplot(probas_negative, label=negative_label, color='red', shade=True)

        # Customizing the plot.
        plt.title(f'Probability Distribution by {positive_label}')
        plt.xticks(np.arange(0, 1, 0.1))
        plt.xlabel('Probabilities')
        plt.ylabel('Density')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.show()
    
    except Exception as e:
        raise CustomException(e, sys)
    

def target_distributions_by_range(y_true, predicted_probas, positive_label='1', negative_label='0'):
    '''
    Plot and analyze the distribution of target classes within score ranges.

    Args:
        y_true (array-like): True target labels.
        predicted_probas (array-like): Predicted probabilities of the positive class.
        positive_label (str, optional): Label for the positive class (default is '1').
        negative_label (str, optional): Label for the negative class (default is '0').

    Returns:
        pandas.DataFrame: DataFrame with columns 'positive_label', 'Scores', and 'Range' showing target flags, scores, and score ranges.

    Raises:
        CustomException: If an error occurs during execution.
    '''

    try:
        # Constructing a dataframe with target flag, scores and range columns.
        bins = np.arange(0, 1.1, 0.1)
        bins_labels = [f'{i/10:.1f} to {i/10+0.1:.1f}' for i in range(10)]

        y_true_scores = pd.DataFrame({positive_label: y_true})
        y_true_scores['Scores'] = predicted_probas
        y_true_scores['Range'] = pd.cut(y_true_scores['Scores'], bins=bins, labels=bins_labels)
        y_true_scores.reset_index(drop=True, inplace=True)
    
        # Plotting the target classes distributions by range.
        plt.figure(figsize=(15, 4))

        # Plotting countplot with hue=positive_label.
        ax = sns.countplot(data=y_true_scores, x='Range', hue=positive_label, palette=['green', 'red'])

        # Customizing the plot.
        plt.title(f'{positive_label} and {negative_label} Distributions by Range')
        plt.xlabel('Range')
        plt.ylabel('Count')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))

        # Adding counts at the top of the bars.
        for p in ax.patches:
            height = p.get_height()
            if not math.isnan(height):
                ax.annotate(f'{round(height)}', (p.get_x() + p.get_width() / 2., height + 0.3), ha='center', va='baseline')

        plt.tight_layout()
        plt.show()

        return y_true_scores

    except Exception as e:
        raise CustomException(e, sys)


def target_percentages_by_range(y_true, predicted_probas, positive_label='1', negative_label='0'):
    '''
    Calculate target percentages by score ranges and visualize them with a stacked bar plot.

    This function takes true labels and predicted probabilities, calculates the target percentages
    within predefined score ranges, and displays them in a stacked bar plot.

    :param y_true: True labels or target values.
    :type y_true: array-like

    :param predicted_probas: Predicted probabilities or scores.
    :type predicted_probas: array-like

    :param positive_label: Label for the positive class (default is '1').
    :type positive_label: str

    :param negative_label: Label for the negative class (default is '0').
    :type negative_label: str

    :return: DataFrame containing percentages of default and non-default within each range.
    :rtype: pandas.DataFrame

    :raises CustomException: If an exception occurs during execution.
    '''
    try:
        # Constructing a dataframe with target flag, scores and range columns.
        bins = np.arange(0, 1.1, 0.1)
        bins_labels = [f'{i/10:.1f} to {i/10+0.1:.1f}' for i in range(10)]

        y_true_scores = pd.DataFrame({positive_label: y_true})
        y_true_scores['Scores'] = predicted_probas
        y_true_scores['Range'] = pd.cut(y_true_scores['Scores'], bins=bins, labels=bins_labels)
        y_true_scores.reset_index(drop=True, inplace=True)

        # Constructing a dataframe with percentages of default and non-default within each range.
        range_percentages = y_true_scores.groupby('Range')['Default'].value_counts(normalize=True).unstack()
        range_percentages_df = range_percentages.rename(columns={0: 'Non-Default', 1: 'Default'})
        range_percentages.reset_index(inplace=True)
        range_percentages.rename(columns={0: 'Non-Default', 1: 'Default'}, inplace=True)
        
        # Creating a stacked bar plot
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Range', y='Default', data=range_percentages, color='green', label='Default')
        sns.barplot(x='Range', y='Non-Default', data=range_percentages, color='red', bottom=range_percentages['Default'], label='Non-Default')

        # Customizing the plot.
        plt.title(f'{positive_label} and {negative_label} Percentages by Range')
        plt.xlabel('Range')
        plt.ylabel('Percentage')
        plt.legend(loc='upper right', bbox_to_anchor=(1, 1))
        plt.xticks(rotation=45)

        # Adding percentages inside the bars.
        for p in range_percentages.index:
            total_height = range_percentages['Default'][p] + range_percentages['Non-Default'][p]
            plt.text(p, range_percentages['Default'][p] / 2, f"{range_percentages['Default'][p]*100:.2f}%", ha='center', color='white', fontsize=10)
            plt.text(p, range_percentages['Default'][p] + range_percentages['Non-Default'][p] / 2, f"{range_percentages['Non-Default'][p]*100:.2f}%", ha='center', color='white', fontsize=10)
        
        plt.tight_layout()
        plt.show()

        return range_percentages_df
    
    except Exception as e:
        raise CustomException(e, sys)