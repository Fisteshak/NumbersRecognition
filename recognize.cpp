#include <vector>
#include <string>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "network.h"

using namespace std;

void readImage(vector <uint8_t>& im, std::string filename) {
    int width, height, bpp;
    uint8_t* rgb_image = stbi_load(filename.c_str(), &width, &height, &bpp, 3);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            im.push_back(255 - (rgb_image[(i * height + j) * 3]));// + rgb_image[i*height + j + 1] + rgb_image[i*height + j + 2]) / 3);
        }
    }

    stbi_image_free(rgb_image);
}


int main() {
    Network net;
    net.loadFromFile("net.data");

    MnistReader reader;
    reader.loadTestingImages();

    std::cout << "                      Welcome to ImageGPT5!\n";
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "To start recognizing draw digit, close paint and press any key.\n";
    std::cout << "To exit type :exit and press enter.\n";
    int succ = net.evaluate(reader.testingImages);
    std::cout << "----------------------------------------------------------------------\n";
    std::cout << "Expected accuracy: " << succ << " / " << reader.testingImages.size() << "\n";
    std::cout << "----------------------------------------------------------------------\n";


    std::cout << "Press any key to continue.\n";;
    std::cin.get();
    while (true) {
        std::string input;
        system("mspaint image.png");
        std::getline(std::cin, input);
        if (input == ":exit") {
            break;
        }
        Image im;
        readImage(im.num, "image.png");
        im.print();

        cout << "result: " << net.getResult(net.feedForward(im.convertToMatrix())) << std::endl;
        cout << "output matrix: \n" << std::fixed << std::setprecision(2) << net.feedForward(im.convertToMatrix()) << std::endl;

    }
}