# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
import math
import random
import time
from collections import defaultdict

import numpy as np

from samples import loadDataFile, loadLabelsFile


# Enhanced Feature Extraction for Digits
def enhanced_digit_feature_extractor(datum):
    features = {}
    pixels = datum.getPixels()
    width, height = datum.width, datum.height

    for x in range(width):
        for y in range(height):
            features[(x, y)] = pixels[x][y]

    for y in range(height):
        for x in range(width // 2):
            features[f'symmetry_{x}_{y}'] = abs(pixels[x][y] - pixels[width - 1 - x][y])

    for x in range(width):
        stroke_density = sum(pixels[x][y] > 0 for y in range(height))
        features[f'stroke_density_x_{x}'] = stroke_density

    for y in range(height):
        stroke_density = sum(pixels[x][y] > 0 for x in range(width))
        features[f'stroke_density_y_{y}'] = stroke_density

    for x in range(1, width - 1):
        for y in range(1, height - 1):
            horiz = pixels[x + 1][y] - pixels[x - 1][y]
            vert = pixels[x][y + 1] - pixels[x][y - 1]
            gradient_magnitude = math.sqrt(horiz**2 + vert**2)
            features[f'gradient_mag_{x}_{y}'] = gradient_magnitude

    min_x = min(y for x in range(width) for y in range(height) if pixels[x][y] > 0)
    max_x = max(y for x in range(width) for y in range(height) if pixels[x][y] > 0)
    min_y = min(x for x in range(width) for y in range(height) if pixels[x][y] > 0)
    max_y = max(x for x in range(width) for y in range(height) if pixels[x][y] > 0)
    bounding_width = max_x - min_x + 1
    bounding_height = max_y - min_y + 1
    features['aspect_ratio'] = bounding_height / (bounding_width + 1e-5)  # Avoid divide-by-zero

    return features

def enhanced_face_feature_extractor(datum):
    features = {}
    pixels = datum.getPixels()
    width, height = datum.width, datum.height

    center_x, center_y = width // 2, height // 2
    for x in range(center_x - 10, center_x + 10):
        for y in range(center_y - 10, center_y + 10):
            if 0 <= x < width and 0 <= y < height:
                features[(x, y)] = pixels[x][y] * 2  
            else:
                features[(x, y)] = pixels[x][y]

    grid_size = 6
    grid_width, grid_height = width // grid_size, height // grid_size
    for gx in range(grid_size):
        for gy in range(grid_size):
            count = 0
            for x in range(gx * grid_width, (gx + 1) * grid_width):
                for y in range(gy * grid_height, (gy + 1) * grid_height):
                    if pixels[x][y] > 0:
                        count += 1
            features[f'grid_{gx}_{gy}'] = count / (grid_width * grid_height)  
            
    return features

def preprocess_digit_data(raw_data):
    return [enhanced_digit_feature_extractor(datum) for datum in raw_data]

def preprocess_face_data(raw_data):
    return [enhanced_face_feature_extractor(datum) for datum in raw_data]

class NaiveBayesClassifier:
    def __init__(self):
        self.feature_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
        self.label_probabilities = defaultdict(float)
        self.labels = []

    def train(self, training_data, training_labels):
        self.labels = list(set(training_labels))
        total_data = len(training_labels)
        label_counts = defaultdict(int)
        feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

        for i, datum in enumerate(training_data):
            label = training_labels[i]
            label_counts[label] += 1
            for key, value in datum.items():
                feature_counts[label][key][value] += 1

        for label in self.labels:
            self.label_probabilities[label] = label_counts[label] / total_data
            for key in feature_counts[label]:
                total_feature_count = sum(feature_counts[label][key].values())
                for value in feature_counts[label][key].keys():
                    self.feature_probabilities[label][key][value] = (
                        feature_counts[label][key][value] + 3) / (total_feature_count + 3 * len(feature_counts[label][key]))

    def predict(self, test_data):
        predictions = []
        for datum in test_data:
            log_probs = {label: math.log(self.label_probabilities[label]) for label in self.labels}
            for label in self.labels:
                for key, value in datum.items():
                    log_probs[label] += math.log(self.feature_probabilities[label][key].get(value, 1e-6))
            predictions.append(max(log_probs, key=log_probs.get))
        return predictions

# Performance Comparison
def evaluate_classifier_performance(classifier, train_data, train_labels, test_data, test_labels, train_percentages, num_trials):
    results = []
    for percent in train_percentages:
        accuracies = []
        errors = []
        training_times = []

        for _ in range(num_trials):
            subset_size = int(len(train_data) * percent / 100)
            indices = random.sample(range(len(train_data)), subset_size)
            subset_train_data = [train_data[i] for i in indices]
            subset_train_labels = [train_labels[i] for i in indices]

            start_time = time.time()
            classifier.train(subset_train_data, subset_train_labels)
            training_time = time.time() - start_time

            predictions = classifier.predict(test_data)
            accuracy = sum(predictions[i] == test_labels[i] for i in range(len(test_labels))) / len(test_labels)
            error = 1 - accuracy

            accuracies.append(accuracy)
            errors.append(error)
            training_times.append(training_time)

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        mean_training_time = np.mean(training_times)

        results.append({
            'percent': percent,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'mean_error': mean_error,
            'std_error': std_error,
            'mean_training_time': mean_training_time
        })

    return results

if __name__ == "__main__":

    raw_digit_train_data = loadDataFile("data/digitdata/trainingimages", 1000, 28, 28)
    digit_train_labels = loadLabelsFile("data/digitdata/traininglabels", 1000)
    raw_digit_test_data = loadDataFile("data/digitdata/testimages", 500, 28, 28)
    digit_test_labels = loadLabelsFile("data/digitdata/testlabels", 500)

    raw_face_train_data = loadDataFile("data/facedata/facedatatrain", 451, 60, 70)
    face_train_labels = loadLabelsFile("data/facedata/facedatatrainlabels", 451)
    raw_face_test_data = loadDataFile("data/facedata/facedatatest", 150, 60, 70)
    face_test_labels = loadLabelsFile("data/facedata/facedatatestlabels", 150)

    digit_train_data = preprocess_digit_data(raw_digit_train_data)
    digit_test_data = preprocess_digit_data(raw_digit_test_data)
    face_train_data = preprocess_face_data(raw_face_train_data)
    face_test_data = preprocess_face_data(raw_face_test_data)


    train_percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    num_trials = 5

    print("\nEvaluating Naive Bayes on Digit Data...")
    classifier = NaiveBayesClassifier()
    digit_results = evaluate_classifier_performance(
        classifier, digit_train_data, digit_train_labels, digit_test_data, digit_test_labels, train_percentages, num_trials)


    print("\nEvaluating Naive Bayes on Face Data...")
    classifier = NaiveBayesClassifier()
    face_results = evaluate_classifier_performance(
        classifier, face_train_data, face_train_labels, face_test_data, face_test_labels, train_percentages, num_trials)


    print("\nDigit Results:")
    for result in digit_results:
        print(f"Training Data: {result['percent']}% - Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} - Mean Error: {result['mean_error']:.4f} ± {result['std_error']:.4f} - Training Time: {result['mean_training_time']:.4f}s")

    print("\nFace Results:")
    for result in face_results:
        print(f"Training Data: {result['percent']}% - Mean Accuracy: {result['mean_accuracy']:.4f} ± {result['std_accuracy']:.4f} - Mean Error: {result['mean_error']:.4f} ± {result['std_error']:.4f} - Training Time: {result['mean_training_time']:.4f}s")
