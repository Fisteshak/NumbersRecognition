#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main()
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    Eigen::Matrix2 <double> n(2, 2);
    n(0, 0) = 0;
    n(0, 1) = 0;
    n(1, 0) = 0;
    n(1, 1) = 0;
    std::cout << n * m << std::endl;

}