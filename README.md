# ML From Scratch to Sklearn

This repository contains implementations of fundamental Machine Learning algorithms from scratch, demonstrating the underlying mathematics and logic before comparing them with scikit-learn implementations.

## 📚 Table of Contents

1. [Regression](#1-regression)
2. [Classification](#2-classification)
3. [Clustering (Unsupervised)](#3-clustering-unsupervised)
4. [Neural Networks](#4-neural-networks)
5. [Getting Started](#-getting-started)
6. [Learning Objectives](#-learning-objectives)

---

## 📚 Algorithms Implemented

### 1. Regression
| Algorithm | Notebook | Key Concepts |
|---|---|---|
| **Linear Regression** | [01_Linear Regression.ipynb](file:///1_Regression/01_Linear%20Regression.ipynb) | Foundations, Optimization |
| **Ridge Regression** | [06_Ridge_Regression_From_Scratch.ipynb](file:///1_Regression/06_Ridge_Regression_From_Scratch.ipynb) | L2 Regularization, Overfitting |
| **Lasso Regression** | [07_Lasso_Regression_From_Scratch.ipynb](file:///1_Regression/07_Lasso_Regression_From_Scratch.ipynb) | L1 Regularization, Feature Selection |
| **Elastic Net** | [08_Elastic_Net_From_Scratch.ipynb](file:///1_Regression/08_Elastic_Net_From_Scratch.ipynb) | L1 + L2 Regularization |
| **Support Vector Regression (SVR)** | [09_SVR_From_Scratch.ipynb](file:///1_Regression/09_SVR_From_Scratch.ipynb) | Epsilon-insensitive loss |
| **Decision Tree Regression** | [10_Decision_Tree_Regression_From_Scratch.ipynb](file:///1_Regression/10_Decision_Tree_Regression_From_Scratch.ipynb) | MSE-based splitting |
| **Random Forest Regression** | [11_Random_Forest_Regression_From_Scratch.ipynb](file:///1_Regression/11_Random_Forest_Regression_From_Scratch.ipynb) | Bagging, Ensembles |
| **Gradient Boosting Regression** | [12_Gradient_Boosting_Regression_From_Scratch.ipynb](file:///1_Regression/12_Gradient_Boosting_Regression_From_Scratch.ipynb) | Sequential learning, Residuals |
| **AdaBoost Regression** | [13_AdaBoost_Regression_From_Scratch.ipynb](file:///1_Regression/13_AdaBoost_Regression_From_Scratch.ipynb) | Weight updates, R2 logic |
| **k-NN Regression** | [14_KNN_Regression_From_Scratch.ipynb](file:///1_Regression/14_KNN_Regression_From_Scratch.ipynb) | Distance-based averaging |

### 2. Classification
| Algorithm | Notebook | Key Concepts |
|---|---|---|
| **Logistic Regression** | [02_Logistic_Regression.ipynb](file:///2_Classification/02_Logistic_Regression_From_Scratch_To_Sklearn.ipynb) | Binary classification, Sigmoid |
| **Decision Trees** | [03_decision tree from scratch.ipynb](file:///2_Classification/03_decision%20tree%20from%20scratch.ipynb) | Entropy, Information Gain |
| **Random Forest (Classifier)** | [04_Random_Forest_From_Scratch.ipynb](file:///2_Classification/04_Random_Forest_From_Scratch.ipynb) | Majority voting, Bootstrapping |
| **Naive Bayes** | [15_Naive_Bayes_From_Scratch.ipynb](file:///2_Classification/15_Naive_Bayes_From_Scratch.ipynb) | Probability, Log-likelihood |
| **k-NN Classification** | [16_KNN_Classification_From_Scratch.ipynb](file:///2_Classification/16_KNN_Classification_From_Scratch.ipynb) | Distance-based voting |
| **Support Vector Machine (SVM)** | [17_SVM_From_Scratch.ipynb](file:///2_Classification/17_SVM_From_Scratch.ipynb) | Max margin, Hinge loss |
| **Gradient Boosting Classifier** | [18_Gradient_Boosting_Classifier_From_Scratch.ipynb](file:///2_Classification/18_Gradient_Boosting_Classifier_From_Scratch.ipynb) | Log-loss optimization |
| **AdaBoost Classifier** | [19_AdaBoost_Classifier_From_Scratch.ipynb](file:///2_Classification/19_AdaBoost_Classifier_From_Scratch.ipynb) | SAMME logic, Stumps |

### 3. Clustering (Unsupervised)
| Algorithm | Notebook | Key Concepts |
|---|---|---|
| **k-Means Clustering** | [20_KMeans_Clustering_From_Scratch.ipynb](file:///3_Clustering/20_KMeans_Clustering_From_Scratch.ipynb) | Centroids, Elbow method |
| **Hierarchical Clustering** | [21_Hierarchical_Clustering_From_Scratch.ipynb](file:///3_Clustering/21_Hierarchical_Clustering_From_Scratch.ipynb) | Dendrograms, Agglomerative |
| **DBSCAN** | [22_DBSCAN_From_Scratch.ipynb](file:///3_Clustering/22_DBSCAN_From_Scratch.ipynb) | Density-based, Outliers |

### 4. Neural Networks
| Algorithm | Notebook | Key Concepts |
|---|---|---|
| **Multi-Layer Perceptron (MLP)** | [05_Neural_Network_From_Scratch.ipynb](file:///4_Neural_Networks/05_Neural_Network_From_Scratch.ipynb) | Backprop, XOR problem |


## 🎯 Learning Objectives

- Understand the mathematical foundations of ML algorithms
- Implement algorithms from scratch using Python and NumPy
- Compare custom implementations with scikit-learn
- Gain intuition about when and how to use different algorithms

### 🚀 Getting Started

#### 1. Environment Setup
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install numpy matplotlib scikit-learn jupyter
```

#### 2. Running the Notebooks
1. Clone this repository
2. Navigate to the project directory
3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Explore the folders and run the notebooks!

---

## 🛤️ Recommended Learning Path

If you are new to Machine Learning, I recommend following this order:
1. **Basics**: `1_Regression/01_Linear Regression.ipynb`
2. **Classification Basics**: `2_Classification/02_Logistic_Regression.ipynb`
3. **Tree Models**: `2_Classification/03_decision tree from scratch.ipynb` -> `04_Random_Forest`
4. **Advanced Regression**: Lasso, Ridge, and Elastic Net
5. **Ensemble Methods**: AdaBoost and Gradient Boosting
6. **Unsupervised**: Start with k-Means clustering
7. **Neural Networks**: Deep dive into the MLP implementation

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
