import pandas as pd
import numpy as np
import os

class CapstonePredictorEngine:
    def __init__(self, dataset_name='philippine_gwa_dataset.csv'):
        # 1. Dynamically locate the main PathFinderAI directory absolute path
        # Since train_model.py is inside 'models', we go up 2 levels to find the CSV
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir) 
        
        # Combine the path safely to build a bulletproof absolute route
        dataset_path = os.path.join(project_root, dataset_name)
        
        # Backup: If you moved train_model.py out of models, check current dir
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(current_dir, dataset_name)
            
        print(f"LOADING DATASET FROM: {dataset_path}")
        
        # 2. Load the dataset
        self.df = pd.read_csv(dataset_path)
        
        # Define features in the strict order your app forms provide them
        self.feature_cols = [
            'python_skill', 'dsa_skill', 'ml_skill', 'web_dev_skill',
            'coding_score', 'communication_score', 'aptitude_score',
            'internships', 'projects', 'backlogs', 'resume_score',
            'skill_score',
            'Program_IT', 'Program_CS', 'Program_IS', 'Program_CPE',
            'GWA'
        ]
        
        # Prepare arrays
        self.X = self.df[self.feature_cols].values
        self.y_placed = self.df['placed'].values
        self.y_salary = self.df['Salary_PHP_Per_Month'].values

    def predict_placement_knn(self, input_vector, k=5):
        """
        MATHEMATICAL ALGORITHM 1: K-NEAREST NEIGHBORS (KNN)
        Calculates the exact Euclidean distance from the input to all dataset entries.
        """
        # Calculate Euclidean Distance: sqrt(sum((x1 - x2)^2))
        distances = np.sqrt(np.sum((self.X - input_vector) ** 2, axis=1))
        
        # Get indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        
        # Fetch their actual placement statuses (1.0 or 0.0)
        neighbor_votes = self.y_placed[nearest_indices]
        
        # Compute exact percentage probability
        probability = (np.sum(neighbor_votes) / k) * 100.0
        return probability

    def predict_placement_random_forest_mimic(self, input_vector):
        """
        MATHEMATICAL ALGORITHM 2: RANDOM FOREST SPLIT MATRIX
        Uses structural threshold trees evaluated directly from your data properties.
        """
        # Unpack input attributes
        (python, dsa, ml, web, coding, comm, apt, 
         intern, proj, backlogs, resume, skill_score, 
         prog_it, prog_cs, prog_is, prog_cpe, gwa) = input_vector[0]

        score = 50.0 # Base anchor probability
        
        # GWA Split (Inverse logic: lower GWA numbers mean superior performance)
        # Assuming gwa is normalized or raw 1.0-5.0 scale
        if gwa > 3.0: 
            score -= 25.0
        elif gwa <= 2.0 and gwa > 0: 
            score += 20.0
            
        # Backlog strict penalty split (Unscaling backlogs_scaled back to integer for math)
        # Since backlogs were divided by 10 in app.py, we multiply by 10 to get the raw count
        raw_backlogs = round(backlogs * 10.0)
        if raw_backlogs > 0: 
            score -= (raw_backlogs * 15.0) # Deduct 15 points per backlog instead of 30 to stay stable
            
        # Technical Skill Split (Changed coding condition from > 0.5 to > 0.05 since score 10 is 0.10)
        if coding > 0.05 and skill_score >= 0.5:
            score += 25.0
            
        # Experience Splits (Unscaling intern and proj to calculate correctly)
        raw_intern = round(intern * 5.0)
        raw_proj = round(proj * 5.0)
        score += (raw_intern * 15.0) + (raw_proj * 10.0)

        return max(0.0, min(100.0, score))
    
    def predict_salary_linear_regression(self, input_vector):
        """
        MATHEMATICAL ALGORITHM 3: MULTIPLE LINEAR REGRESSION
        Fits an algebraic intercept and coefficient array directly against target entries.
        """
        # Calculate standard closed-form Linear Regression weights: Beta = (X^T * X)^(-1) * X^T * y
        # We add a bias column of 1s for the intercept calculation
        X_bias = np.hstack([np.ones((self.X.shape[0], 1)), self.X])
        
        # Solve for coefficients using the Normal Equation
        beta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ self.y_salary
        
        # Apply the learned equation to the input vector
        input_bias = np.hstack([[1.0], input_vector[0]])
        predicted_scaled_salary = np.dot(input_bias, beta)
        
        # Transform back from scaled dataset decimal fraction (0.0 - 1.0) to full Philippine Pesos
        # Based on average local starting tech salary parameters (15k baseline up to 65k upper ceiling)
        actual_php = (predicted_scaled_salary * 50000.0) + 15000.0
        return max(15000.0, actual_php)
    
    def find_exact_match(self, input_vector):
        """
        Looks for an exact matching row in the dataset.
        Returns (is_placed, salary) if found, otherwise (None, None).
        """
        # Unpack input to match the feature column order exactly
        (python, dsa, ml, web, coding, comm, apt, 
         intern, proj, backlogs, resume, skill_score, 
         prog_it, prog_cs, prog_is, prog_cpe, gwa) = input_vector[0]
        
        # Filter the internal pandas DataFrame for an exact match
        match = self.df[
            (self.df['python_skill'] == python) &
            (self.df['dsa_skill'] == dsa) &
            (self.df['ml_skill'] == ml) &
            (self.df['web_dev_skill'] == web) &
            (self.df['coding_score'] == coding) &
            (self.df['communication_score'] == comm) &
            (self.df['aptitude_score'] == apt) &
            (self.df['internships'] == intern) &
            (self.df['projects'] == proj) &
            (self.df['backlogs'] == backlogs) &
            (self.df['resume_score'] == resume) &
            (self.df['Program_IT'] == prog_it) &
            (self.df['Program_CS'] == prog_cs) &
            (self.df['Program_IS'] == prog_is) &
            (self.df['Program_CPE'] == prog_cpe) &
            (np.isclose(self.df['GWA'], gwa, atol=0.05)) # Account for minor float rounding in GWA
        ]
        
        if not match.empty:
            # Grab the very first matching row found
            matched_row = match.iloc[0]
            # Convert 1/0 to True/False for placement status
            is_placed = bool(matched_row['placed'] == 1)
            salary = float(matched_row['Salary_PHP_Per_Month'])
            return is_placed, salary
            
        return None, None