Overview

This project implements and evaluates two machine learning algorithms—Naive Bayes and Perceptron—to classify handwritten digits and facial images. The focus is on analyzing their performance, identifying strengths and weaknesses, and addressing challenges encountered during development. The project demonstrates trade-offs between computational simplicity, accuracy, and efficiency.

Features

Naive Bayes Classifier
	•	Implemented Enhancements:
	•	Laplace Smoothing: Applied with a smoothing parameter of +3 to handle zero probabilities and improve stability.
	•	Feature Engineering: Integrated features like symmetry detection, stroke density, gradient magnitudes, and hole detection using depth-first search (DFS).
	•	Weighted Features: Introduced Gaussian weighting to emphasize central pixels for improved accuracy.
	•	Performance Highlights:
	•	Achieved 71% accuracy for digit classification after extensive tuning.
	•	Excels in face classification with accuracy consistently above 88%.

Perceptron Classifier
	•	Implemented Enhancements:
	•	Weight Optimization: Iteratively adjusted weights using raw pixel intensity values and enhanced features.
	•	Feature Engineering: Incorporated symmetry detection, edge gradients, stroke density, and aspect ratio to capture structural patterns.
	•	Early Stopping: Reduced training time by terminating when no weight updates were required.
	•	Performance Highlights:
	•	Consistently achieved 73% accuracy for digit classification.
	•	Performed exceptionally well for face classification, reaching 89% accuracy.

Getting Started

Prerequisites
	•	Python 3.x
	•	Required libraries: numpy, matplotlib, and scikit-learn (optional for visualization).

Running the Code
	1.	Clone the repository:

git clone <repository_url>
cd digit-face-classification


	2.	Run the classifiers:

python naiveBayes.py
python perceptron.py

Each script outputs a table of results showing performance across different training sizes.

Results Summary

Train Size (%)	Naive Bayes Accuracy	Perceptron Accuracy
10	66%	80%
20	72%	81%
30	76%	82%
40	75%	80%
50	75%	79%
100	76%	81%

Challenges and Solutions

Naive Bayes
	•	Challenge: Numerical instability due to low variances and poor Gaussian likelihood calculation with raw pixel values.
	•	Solution: Introduced Laplace smoothing, Gaussian weighting, and feature engineering for improved stability.

Perceptron
	•	Challenge: Determining optimal iterations and avoiding overfitting.
	•	Solution: Incorporated early stopping and symmetry-based features to enhance convergence and performance.

Conclusion

Both classifiers achieved the goals set by the project:
	•	Naive Bayes: Efficient and interpretable, excelling in face classification but plateauing in digit classification.
	•	Perceptron: Computationally intensive but superior in accuracy and robustness, especially with enhanced features.

These results highlight the trade-offs between simplicity, accuracy, and computational efficiency in machine learning models.
