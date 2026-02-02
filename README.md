# ML From Scratch to Sklearn

This repository contains implementations of fundamental Machine Learning algorithms from scratch, demonstrating the underlying mathematics and logic before comparing them with scikit-learn implementations.

## 📚 Algorithms Implemented

### 1. Linear Regression
- **Notebook**: `01_Linear Regression.ipynb`
- Implementation of linear regression from scratch
- Mathematical foundations and optimization

### 2. Logistic Regression
- **Notebook**: `02_Logistic_Regression_From_Scratch_To_Sklearn.ipynb`
- Binary classification using logistic regression
- From scratch implementation to scikit-learn comparison

### 3. Decision Trees
- **Notebook**: `03_decision tree from scratch.ipynb`
- Entropy and Information Gain calculations
- Tree building algorithm
- Label Encoding vs One-Hot Encoding
- Visualization with scikit-learn

### 4. Random Forest (Classifier)
- **Notebook**: `04_Random_Forest_From_Scratch.ipynb`
- Ensemble learning with bootstrapping
- Feature subset selection
- Majority voting for predictions
- Comparison with single Decision Tree

### 5. Neural Networks (MLP)
- **Notebook**: `05_Neural_Network_From_Scratch.ipynb`
- Multi-Layer Perceptron implementation
- Forward propagation
- Backpropagation algorithm
- Training on XOR problem (non-linear classification)

### 6. Ridge Regression
- **Notebook**: `06_Ridge_Regression_From_Scratch.ipynb`
- L2 Regularization to prevent overfitting
- Closed-form solution implementation

### 7. Decision Tree Regression
- **Notebook**: `10_Decision_Tree_Regression_From_Scratch.ipynb`
- MSE-based splitting for continuous targets
- Tree-based regression logic

### 8. Random Forest Regression
- **Notebook**: `11_Random_Forest_Regression_From_Scratch.ipynb`
- Ensemble of regression trees
- Bootstrapping and averaging results

### 9. k-NN Classification & Regression
- **Notebooks**: `16_KNN_Classification_From_Scratch.ipynb`, `14_KNN_Regression_From_Scratch.ipynb`
- Distance-based learning (Euclidean/Manhattan)
- Majority voting and continuous averaging

### 10. Naive Bayes
- **Notebook**: `15_Naive_Bayes_From_Scratch.ipynb`
- Gaussian, Multinomial, and Bernoulli variants
- Conditional probability and log-likelihood implementation

### 11. Lasso Regression & Elastic Net
- **Notebooks**: `07_Lasso_Regression_From_Scratch.ipynb`, `08_Elastic_Net_From_Scratch.ipynb`
- L1 Regularization and Feature Selection
- Coordinate Descent algorithm implementation

### 12. AdaBoost Classification & Regression
- **Notebooks**: `19_AdaBoost_Classifier_From_Scratch.ipynb`, `13_AdaBoost_Regression_From_Scratch.ipynb`
- Adaptive Boosting with Decision Stumps
- Sample weight updates and SAMME/R2 logic implementation

### 13. Support Vector Machine & Regression
- **Notebooks**: `17_SVM_From_Scratch.ipynb`, `09_SVR_From_Scratch.ipynb`
- Maximum margin classification and regression
- Epsilon-insensitive and Hinge loss implementations

### 14. Gradient Boosting Classification & Regression
- **Notebooks**: `12_Gradient_Boosting_Regression_From_Scratch.ipynb`, `18_Gradient_Boosting_Classifier_From_Scratch.ipynb`
- Sequential ensemble learning
- Residual fitting and log-loss optimization for classification

### 15. Clustering Algorithms (Unsupervised)
- **Notebooks**: `20_KMeans_Clustering_From_Scratch.ipynb`, `21_Hierarchical_Clustering_From_Scratch.ipynb`, `22_DBSCAN_From_Scratch.ipynb`
- centroid-based (k-Means), hierarchy-based (Agglomerative), and density-based (DBSCAN) clustering
- Includes Elbow method and Dendrogram visualizations

## 🎯 Learning Objectives

- Understand the mathematical foundations of ML algorithms
- Implement algorithms from scratch using Python and NumPy
- Compare custom implementations with scikit-learn
- Gain intuition about when and how to use different algorithms

## 🚀 Getting Started

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn jupyter
```

### Running the Notebooks
1. Clone this repository
2. Navigate to the project directory
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Open any notebook and run the cells sequentially

## 📖 Notebook Structure

Each notebook follows a similar structure:
1. **Theory**: Mathematical foundations and concepts
2. **Implementation**: From-scratch implementation with detailed comments
3. **Testing**: Examples and test cases
4. **Comparison**: Scikit-learn implementation and comparison (where applicable)
5. **Visualization**: Plots and visual representations

## 🔍 Key Concepts Covered

- **Supervised Learning**: Linear Regression, Logistic Regression, Decision Trees, Random Forest, Neural Networks
- **Optimization**: Gradient Descent, Information Gain
- **Ensemble Methods**: Bootstrapping, Aggregation (Random Forest)
- **Deep Learning Basics**: Backpropagation, Activation Functions
- **Data Preprocessing**: Label Encoding, One-Hot Encoding

## 📝 Notes

- All implementations prioritize clarity and educational value over performance
- Notebooks include detailed explanations and visualizations
- Code is written to be beginner-friendly while maintaining correctness

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements, bug fixes, or additional algorithms!

## 📄 License

This project is open source and available for educational purposes.
