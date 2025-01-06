# perceptron.py
# -------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import random
import time

import numpy as np

import util
from samples import loadDataFile, loadLabelsFile

PRINT = True


def preprocess_data(raw_data, feature_extractor):
    return [feature_extractor(datum) for datum in raw_data]

class PerceptronClassifier:
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {label: util.Counter() for label in legalLabels}

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = trainingData[0].keys()

        for iteration in range(self.max_iterations):
            updates_made = 0  
            for i, datum in enumerate(trainingData):
                
                scores = {label: self.weights[label] * datum for label in self.legalLabels}
                predicted_label = max(scores, key=scores.get)
                actual_label = trainingLabels[i]

                if predicted_label != actual_label:
                    updates_made += 1
                    self.weights[actual_label] += datum
                    self.weights[predicted_label] -= datum

            if updates_made == 0:
                break

    def classify(self, data):
      
        guesses = []
        for datum in data:
            scores = {label: self.weights[label] * datum for label in self.legalLabels}
            guesses.append(max(scores, key=scores.get))
        return guesses

    def findHighWeightFeatures(self, label):
      
        return self.weights[label].sortedKeys()[:100]
    
    
def enhanced_digit_feature_extractor(datum):
   
    features = {}
    pixels = datum.getPixels()
    width, height = datum.width, datum.height

    
    for x in range(width):
        for y in range(height):
            features[(x, y)] = pixels[x][y]

    
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            horiz = pixels[x + 1][y] - pixels[x - 1][y]
            vert = pixels[x][y + 1] - pixels[x][y - 1]
            features[f'gradient_h_{x}_{y}'] = horiz
            features[f'gradient_v_{x}_{y}'] = vert

    return features

def enhanced_face_feature_extractor(datum):
   
    features = {}
    pixels = datum.getPixels()
    width, height = datum.width, datum.height

   
    for x in range(width):
        for y in range(height):
            features[(x, y)] = pixels[x][y]

    
    center_x, center_y = width // 2, height // 2
    for x in range(center_x - 10, center_x + 10):
        for y in range(center_y - 10, center_y + 10):
            if 0 <= x < width and 0 <= y < height:
                features[(x, y)] *= 2

    return features


def evaluate_perceptron(classifier, train_data, train_labels, test_data, test_labels, train_percentages, num_trials):
 
    results = []
    for percent in train_percentages:
        accuracies = []
        training_times = []

        for _ in range(num_trials):
            
            subset_size = int(len(train_data) * percent / 100)
            indices = random.sample(range(len(train_data)), subset_size)
            subset_train_data = [train_data[i] for i in indices]
            subset_train_labels = [train_labels[i] for i in indices]

          
            start_time = time.time()
            classifier.train(subset_train_data, subset_train_labels, [], [])
            training_time = time.time() - start_time

           
            predictions = classifier.classify(test_data)
            accuracy = sum(predictions[i] == test_labels[i] for i in range(len(test_labels))) / len(test_labels)

            accuracies.append(accuracy)
            training_times.append(training_time)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_error = 1 - mean_accuracy
        mean_training_time = np.mean(training_times)

        results.append({
            'percent': percent,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_error': mean_error,
            'training_time': mean_training_time,
        })

    return results


if __name__ == "__main__":
   
    digit_train_data = loadDataFile("data/digitdata/trainingimages", 1000, 28, 28)
    digit_train_labels = loadLabelsFile("data/digitdata/traininglabels", 1000)
    digit_test_data = loadDataFile("data/digitdata/testimages", 500, 28, 28)
    digit_test_labels = loadLabelsFile("data/digitdata/testlabels", 500)
    
    face_train_data = loadDataFile("data/facedata/facedatatrain", 451, 60, 70)
    face_train_labels = loadLabelsFile("data/facedata/facedatatrainlabels", 451)
    face_test_data = loadDataFile("data/facedata/facedatatest", 150, 60, 70)
    face_test_labels = loadLabelsFile("data/facedata/facedatatestlabels", 150)
   
    digit_train_data = preprocess_data(digit_train_data, enhanced_digit_feature_extractor)
    digit_test_data = preprocess_data(digit_test_data, enhanced_digit_feature_extractor)
   
    face_train_data = preprocess_data(face_train_data, enhanced_face_feature_extractor)
    face_test_data = preprocess_data(face_test_data, enhanced_face_feature_extractor)

    train_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_trials = 5

    print("\nEvaluating Perceptron on Digit Data...")
    digit_classifier = PerceptronClassifier(legalLabels=list(range(10)), max_iterations=10)
    digit_results = evaluate_perceptron(
        digit_classifier, digit_train_data, digit_train_labels, digit_test_data, digit_test_labels, train_percentages, num_trials)

    # Evaluate Perceptron on Faces
    print("\nEvaluating Perceptron on Face Data...")
    face_classifier = PerceptronClassifier(legalLabels=[0, 1], max_iterations=10)
    face_results = evaluate_perceptron(
        face_classifier, face_train_data, face_train_labels, face_test_data, face_test_labels, train_percentages, num_trials)

    # Print Results
    print("\nDigit Results:")
    for result in digit_results:
        print(f"Training Data: {result['percent']}% - Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} - Training Time: {result['training_time']:.4f}s")

    print("\nFace Results:")
    for result in face_results:
        print(f"Training Data: {result['percent']}% - Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} - Training Time: {result['training_time']:.4f}s")



