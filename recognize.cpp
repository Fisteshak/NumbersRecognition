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
        cout << "result:" << net.getResult(net.feedForward(im.convertToMatrix())) << std::endl;
        cout << std::fixed << net.feedForward(im.convertToMatrix()) << std::endl;

    }
}