#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

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

using Eigen::MatrixXd, std::cout, std::cin, std::vector;

std::mt19937_64 random;
const int IMAGE_WIDTH = 28;
const int IMAGE_HEIGHT = 28;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;
const int RESULT_LAYER = 10;
const int LAYERS_NUM = 4;
const std::array <int, LAYERS_NUM> LAYERS_SIZE{ IMAGE_SIZE, 30, 10, RESULT_LAYER };

struct Image {
public:
    vector <unsigned char> num;
    uint8_t label;
    void print() {
        cout << int(label)  << std::endl;
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            for (int j = 0; j < IMAGE_WIDTH; j++) {
                char c;
                if (num[i*IMAGE_WIDTH + j] == 0) c = ' ';
                if (num[i*IMAGE_WIDTH + j] > 0 and num[i*IMAGE_WIDTH + j] <= 100) c = '.';
                if (num[i*IMAGE_WIDTH + j] > 100 and num[i*IMAGE_WIDTH + j] <= 200) c = '*';
                if (num[i*IMAGE_WIDTH + j] > 200) c = '@';

                cout << c << ' ';
            }
            cout << std::endl;
        }
    }


    MatrixXd convertToMatrix() {
        MatrixXd x(IMAGE_SIZE, 1);
        for (int i = 0; i < IMAGE_SIZE; i++) {
            auto t  = double(num[i]) / 255;
            x(i, 0) = t;
          //  x(i, 0)  = num[i] == 0 ? 0 : 1;
        }
        //cout << x;
        return x;
    }
    Image() {};
    Image(vector <unsigned char> num, uint8_t label) : num(num), label(label) {};
    auto operator=(const Image& Im) {return Im;};
    auto operator<=>(const Image&) const = default;
};

struct Network {
public:
    std::function <double(const double)> activationFn = sigmoid;
    std::function <double(const double)> activationDerivativeFn = sigmoidPrime;

    vector <MatrixXd> biases;
    vector <MatrixXd> weights;

    Network() {
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

    MatrixXd feedForward(MatrixXd a) {
        for (int i = 1; i < LAYERS_NUM; i++) {
            a = activationMatrixFn(weights[i] * a + biases[i]);
        }

        return a;
    }
    using TrainingData = vector <Image>;


    void SGD(TrainingData data, int epochs, const int miniBatchSize, const double eta, TrainingData* testData = nullptr) {
        auto rd = std::random_device {};
        auto rng = std::default_random_engine { rd() };
        for (int ep = 0; ep < epochs; ep++) {
            //перемешать

            std::shuffle(std::begin(data), std::end(data), rng);

            for (size_t step = 0; step < data.size(); step += miniBatchSize) {
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
    }

    void updateMiniBatch(TrainingData MiniBatch, double eta) {

        vector <MatrixXd> nabla_w, nabla_b;
        for (auto x : weights) {
            nabla_w.push_back(MatrixXd::Zero(x.rows(), x.cols()));
        }
        for (auto x : biases) {
            nabla_b.push_back(MatrixXd::Zero(x.rows(), x.cols()));
        }

        for (auto x: MiniBatch) {

            auto [delta_nabla_b, delta_nabla_w] = backprop(x);

            for (size_t i = 0; i <nabla_b.size(); i++) {
                nabla_b[i] = nabla_b[i] + delta_nabla_b[i];
            }
            for (size_t i = 0; i < nabla_w.size(); i++) {
                nabla_w[i] = nabla_w[i] + delta_nabla_w[i];
            }

            for (size_t i = 0; i < nabla_b.size(); i++) {
                biases[i] = biases[i] - (eta / MiniBatch.size())*nabla_b[i];
            }
            for (size_t i = 0; i < nabla_w.size(); i++) {
                weights[i] = weights[i] - (eta / MiniBatch.size())*nabla_w[i];
            }
        }
    }

    std::tuple <vector<MatrixXd>, vector<MatrixXd>> backprop(Image im) {
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
        return {nabla_b, nabla_w};
    }

    int evaluate(TrainingData testData) {
        vector <bool> res;

        for (auto x: testData) {
            auto ind = getResult(feedForward(x.convertToMatrix()));
            res.push_back(ind == x.label);
        }
        int sum = 0;
        for (auto x: res) {
            sum += x;
        }
        return sum;
    }
    int getResult(MatrixXd output) {
        int ind = 0;
        for (int i = 1; i < output.rows(); i++) {
            if (output(i, 0) > output(ind, 0))
                ind = i;
        }
        return ind;
    }

};


struct MnistReader {
private:


    int reverseInt(const int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    }

    void loadNumbers(const std::string& filename, vector<Image>& images) {
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

    void loadLabels(const std::string& filename, vector<Image>& images) {
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

    void loadImages(const std::string& filename_numbers, const std::string& filename_labels, vector<Image>& images) {
        loadNumbers(filename_numbers, images);
        loadLabels(filename_labels, images);
    }

public:
    vector <Image> trainingImages;
    vector <Image> testingImages;


    void loadTrainingImages() {
        loadImages("train-images.idx3-ubyte", "train-labels.idx1-ubyte", trainingImages);
    }

    void loadTestingImages() {
        loadImages("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", testingImages);
    }

    void printImage(const Image& im) {
        cout << (int)im.label << '\n';
        for (int i = 0; i < IMAGE_HEIGHT; i++) {
            for (int j = 0; j < IMAGE_WIDTH; j++) {
                cout << (im.num[i * IMAGE_HEIGHT + j] == 0 ? ' ' : '*') << ' ';
            }
            cout << '\n';
        }
    }
};

void readImage(vector <uint8_t> &im, std::string filename) {
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(filename.c_str(), &width, &height, &bpp, 3);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            im.push_back(255 - (rgb_image[(i*height + j)*3]));// + rgb_image[i*height + j + 1] + rgb_image[i*height + j + 2]) / 3);
        }
    }

    stbi_image_free(rgb_image);
}

int main() {

    srand(time(nullptr));
    MnistReader mnistReader;

    mnistReader.loadTrainingImages();
    mnistReader.loadTestingImages();

    Network net;
    net.SGD(mnistReader.trainingImages, 5, 30, 0.2, &mnistReader.testingImages);

    int successNum = net.evaluate(mnistReader.testingImages);
    cout << "Accuracy: " << (double)successNum / mnistReader.testingImages.size() * 100 << "%\n";

    for (int i = 0; i < 10; i++) {

        system("mspaint image.png");
        cin.get();
        //Image im = mnistReader.testingImages[i];
        Image im;
        readImage(im.num, "image.png");
        //im.label = -1;
        im.print();
        cout << "result:" << net.getResult(net.feedForward(im.convertToMatrix())) << std::endl;
        cout << std::fixed  << net.feedForward(im.convertToMatrix()) << std::endl;

    }

}
