# Нейронная сеть, распознающая цифры на основе MNIST
## Компиляция:
`g++ -I path_to_eigen/eigen  recognize.cpp network.cpp -o recognize.exe -std=c++20 -O3 -march=native`

`g++ -I path_to_eigen/eigen  learn.cpp network.cpp -o learn.exe -std=c++20 -O3 -march=native`
## Запуск:
`./learn.exe` - обучает нейросеть и сохраняет веса в net.data

`./recognize.exe` - распознавание цифр

Программа принимает изображения 28x28 пикселей в формате png. Сами цифры должно быть размером не больше 20x20 пикселей и расположены по центру, иначе распознавание работает крайне плохо!

