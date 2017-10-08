#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse(4);
    rmse << 0,0,0,0;

    if(estimations.size() != ground_truth.size()
                || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    for (int i=0; i<(int)estimations.size(); i++) {

        for (int j=0; j<(int)estimations[i].size(); j++) {
            float dif = estimations[i](j) - ground_truth[i](j);
            rmse(j) += dif*dif;
        }
    }

    rmse = rmse/estimations.size();

    rmse = rmse.array().sqrt();

    return rmse;
}

MatrixXd & Tools::CalculateJacobian(const VectorXd& x_state) {

    static MatrixXd Hj(3,4);

    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    float normsqr = px*px + py*py;
    float norm = sqrt(normsqr);

    if (normsqr > 1.0e-6) {
        Hj(0, 0) = Hj(2, 2) = px / norm;
        Hj(0, 1) = Hj(2, 3) = py / norm;
        Hj(1, 0) = -py / normsqr;
        Hj(1, 1) = px / normsqr;

        Hj(2, 0) = py*(vx*py - vy*px) / pow(normsqr, 1.5);
        Hj(2, 1) = -px*(vx*py - vy*px) / pow(normsqr, 1.5);
    }
    else {
        cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    }

    return Hj;


}
