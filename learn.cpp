#include "network.h"

#include <iostream>

using namespace std;

int main() {

    srand(time(nullptr));
    MnistReader mnistReader;

    mnistReader.loadTrainingImages();
    mnistReader.loadTestingImages();

    Network net;
    net.SGD(mnistReader.trainingImages, 10, 10, 0.1, &mnistReader.testingImages);

    int successNum = net.evaluate(mnistReader.testingImages);
    cout << "Accuracy: " << (double)successNum / mnistReader.testingImages.size() * 100 << "%\n";
    net.loadToFile("net.data");


}