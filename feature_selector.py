import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from skrebate import ReliefF
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from itertools import combinations
import logging
import os
import warnings

warnings.filterwarnings('ignore')

dataset = pd.read_csv("/home/aghasemi/CompBio481/feat_select/feat_select/feat_slct_bin.csv")

class FeatureSelector:
    def __init__(self, data, target_column, out_folder):
        """
        Initialize the feature selector.

        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
        - out_folder: Directory where output files will be saved.
        """
        self.data = data
        self.target_column = target_column
        self.out_folder = out_folder

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        logging.basicConfig(filename=os.path.join(out_folder, 'feature_selector.log'), level=logging.DEBUG)

    def t_test(self, data, target_column):
        """
        Perform a t-test for each feature against the target.

        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.

        Returns:
        - results: DataFrame containing the t-statistic and p-value for each feature.
        """
        results = pd.DataFrame(columns=['Feature', 'T-Statistic', 'P-Value'])
        target = data[target_column]
        for feature in data.drop(columns=[target_column, 'ID_1']).columns:
            group1 = data[data[target_column] == 0][feature]
            group2 = data[data[target_column] == 1][feature]
            t_stat, p_val = ttest_ind(group1, group2, nan_policy='omit')
            results = results.append({'Feature': feature, 'T-Statistic': t_stat, 'P-Value': p_val}, ignore_index=True)
        return results

    def mutual_score(self, data, target_column):
        """
        Compute Mutual Information Score for each feature against the target.

        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.

        Returns:
        - results: DataFrame containing Mutual Information Score for each feature.
        """
        features = data.drop(columns=[target_column, 'ID_1']).values
        target = data[target_column].values
        mi_scores = mutual_info_classif(features, target)
        feature_names = data.drop(columns=[target_column, 'ID_1']).columns
        results = pd.DataFrame({'Feature': feature_names, 'Mutual Information Score': mi_scores})
        return results

    def relieff(self, data, target_column):
        """
        Apply the ReliefF algorithm to rank features based on their importance.
    
        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
    
        Returns:
        - results: DataFrame containing features ranked by their ReliefF score.
        """
        # Extract feature values and target values from the DataFrame
        features = data.drop(columns=[target_column, 'ID_1']).values  # Exclude target and ID columns
        target = data[target_column].values  # Target values
    
        # Initialize the ReliefF algorithm with the specified number of neighbors
        fs = ReliefF(n_neighbors=100)  # Number of neighbors can be adjusted based on dataset size
    
        # Fit the ReliefF model to the data
        fs.fit(features, target)
    
        # Retrieve feature importance scores from the fitted model
        scores = fs.feature_importances_
    
        # Map the ReliefF scores to the corresponding feature names
        feature_names = data.drop(columns=[target_column, 'ID_1']).columns
        results = pd.DataFrame({'Feature': feature_names, 'ReliefF Score': scores})
    
        # Return the DataFrame containing features and their ReliefF scores
        return results

    def pearson_correlation(self, data, target_column):
        """
        Compute Pearson's Correlation Coefficient for each feature against the target.
    
        Arguments:
        - data: DataFrame containing the features and target.
        - target_column: The name of the target column in the data.
    
        Returns:
        - results: DataFrame containing Pearson's Correlation Coefficient for each feature.
        """
        features = data.drop(columns=[target_column, 'ID_1'])
        correlations = features.corrwith(data[target_column])
        results = pd.DataFrame({'Feature': correlations.index, 'Pearson Correlation': correlations.values})
        return results

    def variance_threshold(self, data, threshold=0.01):
        """
        Select features based on variance threshold.
    
        Arguments:
        - data: DataFrame containing the features and target.
        - threshold: Features with a variance lower than this threshold will be removed.
    
        Returns:
        - results: DataFrame containing the remaining features after applying the variance threshold.
        """
        # Separate features from the target variable
        features = data.drop(columns=[self.target_column, 'ID_1'])
        
        # Initialize and fit the VarianceThreshold
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(features)
        
        # Get the mask of features that meet the variance threshold
        features_mask = selector.get_support(indices=True)
        
        # Apply the mask to retain only the selected features
        selected_features = features.iloc[:, features_mask]
        
        # Create a DataFrame to return the results
        results = pd.DataFrame(selected_features.columns, columns=['Feature'])
        results['Variance'] = selector.variances_[features_mask]
        
        return results

    def bootstrap_feature_selection(self, n_bootstraps, n_samples=None, stratify=True, tests=['ttest', 'relieff', 'mutual_score']):
        """
        Perform bootstrapped feature selection for specified tests.

        Arguments:
        - n_bootstraps: Number of bootstrap samples to create.
        - n_samples: Number of samples in each bootstrap. Default is the size of the original data.
        - stratify: Whether to stratify the bootstrap samples based on the target column.
        - tests: List of tests to perform. Options: 'ttest', 'relieff', 'mutual_score'.
        """
        for i in range(n_bootstraps):
            if stratify:
                bootstrap_sample = resample(self.data, n_samples=n_samples, stratify=self.data[self.target_column], replace=True)
            else:
                bootstrap_sample = resample(self.data, n_samples=n_samples, replace=True)
            
            for test in tests:
                if test == 'ttest':
                    test_results = self.t_test(bootstrap_sample, self.target_column)
                elif test == 'relieff':
                    test_results = self.relieff(bootstrap_sample, self.target_column)
                elif test == 'mutual_score':
                    test_results = self.mutual_score(bootstrap_sample, self.target_column)
                elif test == 'pearson_corr':
                    test_results = self.pearson_correlation(bootstrap_sample, self.target_column)
                elif test == 'variance_thres':
                    test_results = self.variance_threshold(bootstrap_sample, threshold=0.01)
                
                test_results.to_csv(os.path.join(self.out_folder, f'bootstrap_{i+1}_{test}_results.csv'), index=False)


def main():
    dataset = pd.read_csv("/home/aghasemi/CompBio481/feat_select/feat_select/feat_slct_bin.csv")
    fs = FeatureSelector(dataset, 'Diagnosis', '/home/aghasemi/CompBio481/feat_select/feat_select/feat_select_res')
    # Specify the tests you want to run in the list, e.g., ['ttest', 'relieff', 'mutual_score', 'pearson_corr', 'variance_thres']
    fs.bootstrap_feature_selection(n_bootstraps=1, n_samples=500, stratify=True, tests=['variance_thres'])

if __name__ == "__main__":
    main()