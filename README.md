# PathFinder-AI
Multi-Stage Machine Learning Predictive Placement and Salary Analytics Engine.

PathFinder AI is a full-stack predictive web application designed to forecast academic graduate outcomes, recommend tailored career positions, and estimate starting salary variables. The engine integrates a React/Vite/TypeScript frontend with an active Python Flask API backend running synchronized scikit-learn models.

## Core Architecture Blueprint

The application employs a 3-Stage Pipeline executing linear, spatial, and probabilistic calculations:
1. **Stage 1 (Classification):** A Random Forest Classifier determines historical deployment probability based on academic marks (GWA), internship parameters, and active backlog indices.
2. **Stage 2 (Regression):** A Multiple Linear Regression model dynamically scales entry-level salary compensation forecasts matching corporate tier environments.
3. **Stage 3 (Spatial Optimization):** A K-Nearest Neighbors (KNN) model assigns optimal career paths based on localized skill alignment across core domains (Python, DSA, Web Dev, ML).

