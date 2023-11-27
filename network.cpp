
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
#include "network.h"

using Eigen::MatrixXd, std::cout, std::cin, std::vector;

std::mt19937_64 random;

void Image::print() const {
    //cout << int(label) << std::endl;
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            char c;
            if (num[i * IMAGE_WIDTH + j] == 0) c = ' ';
            if (num[i * IMAGE_WIDTH + j] > 0 and num[i * IMAGE_WIDTH + j] <= 100) c = '.';
            if (num[i * IMAGE_WIDTH + j] > 100 and num[i * IMAGE_WIDTH + j] <= 200) c = '*';
            if (num[i * IMAGE_WIDTH + j] > 200) c = '@';

            cout << c << ' ';
        }
        cout << std::endl;
    }
}

MatrixXd Image::convertToMatrix() const {
    MatrixXd x(IMAGE_SIZE, 1);
    for (int i = 0; i < IMAGE_SIZE; i++) {
        auto t = double(num[i]) / 255;
        x(i, 0) = t;
    }
    return x;
}


int MnistReader::reverseInt(const int i) const {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void MnistReader::loadNumbers(const std::string& filename, vector<Image>& images) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        cout << filename << " file not found!\n";
        exit(-1);
    }

    int magicNumber;
    int numberOfImages;
    int numberOfRows;
    int numberOfColumns;
    fin.read((char*)&magicNumber, 4);
    fin.read((char*)&numberOfImages, 4);
    fin.read((char*)&numberOfRows, 4);
    fin.read((char*)&numberOfColumns, 4);

    magicNumber = reverseInt(magicNumber);
    numberOfImages = reverseInt(numberOfImages);
    numberOfRows = reverseInt(numberOfRows);
    numberOfColumns = reverseInt(numberOfColumns);

    images.resize(numberOfImages);
    for (int i = 0; i < numberOfImages; i++) {
        images[i].num.assign(IMAGE_SIZE, 0);
        fin.read((char*)images[i].num.data(), IMAGE_SIZE);
    }
}

void MnistReader::loadLabels(const std::string& filename, vector<Image>& images) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin.is_open()) {
        cout << filename << " file not found!\n";
        exit(-1);
    }

    int magicNumber;
    int numberOfLabels;
    fin.read((char*)&magicNumber, 4);
    fin.read((char*)&numberOfLabels, 4);

    magicNumber = reverseInt(magicNumber);
    numberOfLabels = reverseInt(numberOfLabels);

    for (int i = 0; i < numberOfLabels; i++) {
        fin.read((char*)&images[i].label, 1);
    }

}


void MnistReader::loadImages(const std::string& filename_numbers, const std::string& filename_labels, vector<Image>& images) {
    loadNumbers(filename_numbers, images);
    loadLabels(filename_labels, images);
}

void MnistReader::loadTrainingImages() {
    loadImages("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte", trainingImages);
}

void MnistReader::loadTestingImages() {
    loadImages("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte", testingImages);
}

void MnistReader::printImage(const Image& im)  const{
    cout << (int)im.label << '\n';
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            cout << (im.num[i * IMAGE_HEIGHT + j] == 0 ? ' ' : '*') << ' ';
        }
        cout << '\n';
    }
}


Network::Network() {
    biases.resize(LAYERS_NUM);
    weights.resize(LAYERS_NUM);

    for (int i = 1; i < LAYERS_NUM; i++) {
        biases[i] = MatrixXd::Random(LAYERS_SIZE[i], 1);
        weights[i] = MatrixXd::Random(LAYERS_SIZE[i], LAYERS_SIZE[i - 1]);
    }
}

MatrixXd costDerivative(MatrixXd outputActivations, MatrixXd y) {
    assert(outputActivations.cols() == y.cols());
    assert(outputActivations.rows() == y.rows());

    return outputActivations - y;
}


MatrixXd Network::feedForward(MatrixXd a) {
    for (int i = 1; i < LAYERS_NUM; i++) {
        a = activationMatrixFn(weights[i] * a + biases[i]);
    }

    return a;
}


void Network::SGD(TrainingData data, int epochs, const int miniBatchSize, const double eta, TrainingData* testData = nullptr) {
    auto rd = std::random_device{};
    auto rng = std::default_random_engine{ rd() };
    for (int ep = 0; ep < epochs; ep++) {
        //перемешать

        std::shuffle(std::begin(data), std::end(data), rng);

        for (size_t step = 0; step + miniBatchSize < data.size(); step += miniBatchSize) {
            //выделить кусок
            TrainingData miniBatch(miniBatchSize);
            for (int i = 0; i < miniBatchSize; i++) {
                miniBatch[i].label = data[step + i].label;
                miniBatch[i].num = data[step + i].num;
            }

            updateMiniBatch(miniBatch, eta);
        }

        if (testData != nullptr) {
            int succ = evaluate(*testData);
            cout << "Epoch: " << ep << " " << succ << " / " << testData->size() << "\n";
        }
    }
    return;
}

void Network::updateMiniBatch(TrainingData MiniBatch, double eta) {

    vector <MatrixXd> nabla_w, nabla_b;
    for (auto x : weights) {
        nabla_w.push_back(MatrixXd::Zero(x.rows(), x.cols()));
    }
    for (auto x : biases) {
        nabla_b.push_back(MatrixXd::Zero(x.rows(), x.cols()));
    }

    for (auto x : MiniBatch) {

        auto [delta_nabla_b, delta_nabla_w] = backprop(x);

        for (size_t i = 0; i < nabla_b.size(); i++) {
            nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
        }
        for (size_t i = 0; i < nabla_w.size(); i++) {
            nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
        }

        for (size_t i = 0; i < nabla_b.size(); i++) {
            biases[i] = biases[i] - (eta / MiniBatch.size()) * nabla_b[i];
        }
        for (size_t i = 0; i < nabla_w.size(); i++) {
            weights[i] = weights[i] - (eta / MiniBatch.size()) * nabla_w[i];
        }
    }
    return;
}

std::tuple <vector<MatrixXd>, vector<MatrixXd>> Network::backprop(Image im) {
    vector <MatrixXd> nabla_w, nabla_b;
    for (auto x : weights) {
        nabla_w.push_back(MatrixXd::Zero(x.rows(), x.cols()));
    }
    for (auto x : biases) {
        nabla_b.push_back(MatrixXd::Zero(x.rows(), x.cols()));
    }
    vector <MatrixXd> activations, zs;
    auto activation = im.convertToMatrix();
    activations.push_back(activation);

    for (int i = 1; i < LAYERS_NUM; i++) {
        MatrixXd z = weights[i] * activation + biases[i];
        zs.push_back(z);
        activation = activationMatrixFn(z);
        activations.push_back(activation);
    }

    MatrixXd expectedOuput = MatrixXd::Zero(RESULT_LAYER, 1);
    expectedOuput(im.label, 0) = 1;


    MatrixXd delta = costDerivative(activations.back(), expectedOuput).cwiseProduct(activationDerivativeMatrixFn(zs.back()));
    nabla_b.back() = delta;
    nabla_w.back() = delta * activations[activations.size() - 2].transpose();
    for (int l = 2; l < LAYERS_NUM; l++) {
        MatrixXd z = zs[zs.size() - l];
        MatrixXd sp = activationDerivativeMatrixFn(z);
        delta = (weights[weights.size() - l + 1].transpose() * delta).cwiseProduct(sp);
        nabla_b[nabla_b.size() - l] = delta;
        nabla_w[nabla_w.size() - l] = delta * activations[activations.size() - l - 1].transpose();
    }
    return { nabla_b, nabla_w };
}

int Network::evaluate(TrainingData testData) {
    std::vector<bool> res(testData.size());

    std::transform(testData.begin(), testData.end(), res.begin(), [this](auto& x) {
        auto ind = getResult(feedForward(x.convertToMatrix()));
        return ind == x.label;
    });

    int sum = std::accumulate(res.begin(), res.end(), 0);

    return sum;
}
int Network::getResult(MatrixXd output) const {
    Eigen::Index maxRow, maxCol;
    output.maxCoeff(&maxRow, &maxCol);
    return maxRow;
}

void Network::loadToFile(std::string filename) {
    std::ofstream fout(filename);
    fout << biases.size() << ' ' << weights.size() << '\n';
    for (auto x: biases) {
        fout << x.rows() << ' ' << x.cols() << '\n';
        fout << x << '\n';
    }
    for (auto x: weights) {
        fout << x.rows() << ' ' << x.cols() << '\n';
        fout << x << '\n';
    }
}

void Network::loadFromFile(std::string filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        cout << filename << " file does not exist!\n";
        exit(0);
    }
    size_t biasesSize, weightsSize;
    fin >> biasesSize >> weightsSize;
    biases.resize(biasesSize);
    weights.resize(weightsSize);
    for (auto& x : biases) {
        Eigen::Index rows, cols;
        fin >> rows >> cols;
        x.resize(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fin >> x(i, j);
            }
        }
    }
    for (auto& x : weights) {
        Eigen::Index rows, cols;
        fin >> rows >> cols;
        x.resize(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                fin >> x(i, j);
            }
        }

    }
}






