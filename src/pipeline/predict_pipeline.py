'''
This script aims to create the predict pipeline for a simple web application which will be interacting with the pkl files, such that we can make predictions by giving values of input features. 
'''

# Debugging and verbose.
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object

# Data manipulation.
import pandas as pd

# File handling.
import os


class PredictPipeline:
    '''
    Class for making predictions using a trained model and preprocessor.

    This class provides a pipeline for making predictions on new instances using a trained machine learning model and
    a preprocessor. It loads the model and preprocessor from files, preprocesses the input features, and applies a
    predefined threshold for making predictions.

    Methods:
        predict(features):
            Make predictions on new instances using the loaded model and preprocessor.

    Example:
        pipeline = PredictPipeline()
        new_features = [...]
        prediction = pipeline.predict(new_features)

    Note:
        This class assumes the availability of the load_object function and defines the THRESHOLD value.
    '''
    def __init__(self) -> None:
        '''
        Initializes a PredictPipeline instance.

        Initializes the instance. No specific setup is required in the constructor.
        '''
        pass


    def predict(self, features):
        '''
        Make predictions on new instances using the loaded model and preprocessor.

        Args:
            features: Input features for which predictions will be made.

        Returns:
            predictions: Predicted labels for the input features.

        Raises:
            CustomException: If an exception occurs during the prediction process.
        
        Note: The predictions are made by comparing the model's estimated probabilities of the customer being bad risk with a threshold value. If the probability is greater than it, the instance is classified as positive, else negative. The threshold was chosen during modelling notebook, where I saw that its value provides a recall score of 0.8 (the metric of interest).
        '''
        try:


            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            
            logging.info('Load model and preprocessor objects.')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info('Model and preprocessor succesfully loaded.')

            logging.info('Mapping sex variable.')

            # Sex variable mapping needed.
            features['sex'] = features['sex'].map({'male': 1, 'female': 0})

            logging.info('Preprocessing the input data.')

            prepared_data = preprocessor.transform(features)

            logging.info('Input data prepared for prediction.')

            logging.info('Predicting.')

            # Predict the customer's probability of defaulting on a loan and assigning him a credit score.
            default_proba = model.predict_proba(prepared_data)[:, 1]

            if default_proba >= 0.0 and default_proba < 0.1:
                credit_score = "Customer's credit score: 1000"
            elif default_proba >= 0.1 and default_proba < 0.2:
                credit_score = "Customer's credit score: 900"
            elif default_proba >= 0.2 and default_proba < 0.3:
                credit_score = "Customer's credit score: 800"
            elif default_proba >= 0.3 and default_proba < 0.4:
                credit_score = "Customer's credit score: 700"
            elif default_proba >= 0.4 and default_proba < 0.5:
                credit_score = "Customer's credit score: 600"
            elif default_proba >= 0.5 and default_proba < 0.6:
                credit_score = "Customer's credit score: 500"
            elif default_proba >= 0.6 and default_proba < 0.7:
                credit_score = "Customer's credit score: 400"
            elif default_proba >= 0.7 and default_proba < 0.8:
                credit_score = "Customer's credit score: 300"
            elif default_proba >= 0.8 and default_proba < 0.9:
                credit_score = "Customer's credit score: 200"
            elif default_proba >= 0.9 and default_proba < 1:
                credit_score = "Customer's credit score: 100"
            
            logging.info('Default probability successfully estimated. Credit score successfully assigned.')

            return credit_score

        except Exception as e:
            raise CustomException(e, sys)
        

class InputData:
    '''
    Class for handling input data for predictions.

    This class provides a structured representation for input data that is meant to be used for making predictions.
    It maps input variables from HTML inputs to class attributes and provides a method to convert the input data into
    a DataFrame format suitable for making predictions.

    Attributes:
        age (int): Age of the individual.
        sex (str): Gender of the individual.
        job (int): Job category of the individual.
        housing (str): Housing situation of the individual.
        saving_accounts (str): Type of saving accounts.
        checking_account (str): Type of checking account.
        credit_amount (int): Amount of credit.
        duration (int): Duration of the credit.
        purpose (str): Purpose of the credit.

    Methods:
        get_input_data_df():
            Convert the mapped input data into a DataFrame for predictions.

    Example:
        input_instance = InputData(age=30, sex='male', job=2, housing='own', saving_accounts='little',
                                   checking_account='moderate', credit_amount=5000, duration=12, purpose='car')
        input_df = input_instance.get_input_data_df()

    Note:
        This class assumes the availability of the pandas library and defines the CustomException class.
    '''

    def __init__(self,
                 age: int,
                 sex: str,
                 job: int,
                 housing: str,
                 saving_accounts: str,
                 checking_account: str,
                 credit_amount: int,
                 duration: int,
                 purpose: str) -> None:
        '''
        Initialize an InputData instance with mapped input data.

        Args:
            age (int): Age of the individual.
            sex (str): Gender of the individual.
            job (int): Job category of the individual.
            housing (str): Housing situation of the individual.
            saving_accounts (str): Type of saving accounts.
            checking_account (str): Type of checking account.
            credit_amount (int): Amount of credit.
            duration (int): Duration of the credit.
            purpose (str): Purpose of the credit.
        '''
        
        # Map variables from html inputs.
        self.age = age
        self.sex = sex
        self.job = job
        self.housing = housing
        self.saving_accounts = saving_accounts
        self.checking_account = checking_account
        self.credit_amount = credit_amount
        self.duration = duration
        self.purpose = purpose

    
    def get_input_data_df(self):
        '''
        Convert the mapped input data into a DataFrame for predictions.

        Returns:
            input_data_df (DataFrame): DataFrame containing the mapped input data.

        Raises:
            CustomException: If an exception occurs during the process.
        '''
        try:
            input_data_dict = dict()

            # Map the variables to the form of a dataframe for being used in predictions.
            
            input_data_dict['age'] = [self.age]
            input_data_dict['sex'] = [self.sex]
            input_data_dict['job'] = [self.job]
            input_data_dict['housing'] = [self.housing]
            input_data_dict['saving_accounts'] = [self.saving_accounts]
            input_data_dict['checking_account'] = [self.checking_account]
            input_data_dict['credit_amount'] = [self.credit_amount]
            input_data_dict['duration'] = [self.duration]
            input_data_dict['purpose'] = [self.purpose]

            input_data_df = pd.DataFrame(input_data_dict)

            return input_data_df
        
        except Exception as e:
            raise CustomException(e, sys)