#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <algorithm>
#include <compare>
#include <tuple>
#include <iomanip>
#include <functional>
#include <cmath>
#include <numbers>
#include <string>


const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const int RESULT_LAYER = 10;
const int LAYERS_NUM = 4;
const std::array <int, LAYERS_NUM> LAYERS_SIZE{ IMAGE_SIZE, 100, 30, RESULT_LAYER };
using Eigen::MatrixXd, std::cout, std::cin, std::vector;

struct Image {
public:
    vector <unsigned char> num;
    uint8_t label;
    void print() const;
    MatrixXd convertToMatrix() const;
    Image() {};
    Image(vector <unsigned char> num, uint8_t label) : num(num), label(label) {};
    auto operator=(const Image& Im) {return Im;};
    auto operator<=>(const Image&) const = default;
};

struct MnistReader {
private:
    int reverseInt(const int i) const;
    void loadNumbers(const std::string& filename, vector<Image>& images);
    void loadLabels(const std::string& filename, vector<Image>& images);
    void loadImages(const std::string& filename_numbers, const std::string& filename_labels, vector<Image>& images);

public:
    vector <Image> trainingImages;
    vector <Image> testingImages;
    void loadTrainingImages();
    void loadTestingImages();
    void printImage(const Image& im) const;
};


struct Network {
public:
    using TrainingData = vector <Image>;

    Network();

    void SGD(TrainingData data, int epochs, const int miniBatchSize, const double eta, TrainingData* testData);
    int evaluate(TrainingData testData);
    MatrixXd feedForward(MatrixXd a);
    int getResult(MatrixXd output) const;
    void loadToFile(std::string filename);
    void loadFromFile(std::string filename);


private:
    std::function <double(const double)> activationFn = sigmoid;
    std::function <double(const double)> activationDerivativeFn = sigmoidPrime;
    vector <MatrixXd> biases;
    vector <MatrixXd> weights;

    MatrixXd costDerivative(MatrixXd outputActivations, MatrixXd y) {
        return outputActivations - y;
    }

    static double RELU(const double x) {
        return x > 0 ? x / 10 : 0;
    }
    static double RELUDerivative(const double x) {
        return x > 0 ? 0.1 : 0;
    }

    static double sigmoid(const double x) {
        return 1.0 / (1.0 + exp(-x));
    }
    static double sigmoidPrime(const double x) {
        return sigmoid(x) * (1 - sigmoid(x));
    }

    static double tanh(const double x) {
        return std::tanh(x);
    }
    static double tanhDerivative(const double x) {
        return 1 - std::tanh(x) * std::tanh(x);
    }

    MatrixXd activationMatrixFn(MatrixXd z) {
        return z.unaryExpr(activationFn);
    }

    MatrixXd activationDerivativeMatrixFn(MatrixXd z) {
        return z.unaryExpr(activationDerivativeFn);
    }

    void updateMiniBatch(TrainingData MiniBatch, double eta);
    std::tuple <vector<MatrixXd>, vector<MatrixXd>> backprop(Image im);
};




