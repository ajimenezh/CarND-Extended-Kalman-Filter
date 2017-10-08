#include "kalman_filter.h"
#include "tools.h"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
  I_ = MatrixXd::Identity(4, 4);
  for (int i=0; i<4; i++) I_(i, i) = 1;
}

void KalmanFilter::Predict() {

    //cout<<"Predict"<<endl;
    //cout<<x_<<endl<<F_<<endl;

    x_ = F_ * x_ /*+ u*/;
    MatrixXd Ft = F_.transpose();

    P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

    //cout<<"Start update"<<endl;

    VectorXd y = z - H_ * x_;
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    x_ = x_ + (K * y);
    P_ = (I_ - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

    //cout<<"Start update EKF"<<endl;

    MatrixXd Hj = Tools::CalculateJacobian(x_);

    VectorXd y = z;

    y(0) -= sqrt(x_(0)*x_(0) + x_(1)*x_(1));
    y(1) -= atan2(x_(1), x_(0));

    static float pi = acos(-1.0);

    while (y(1) > pi) {
        y(1) -= 2*pi;
    }
    while (y(1) < -pi) {
        y(1) += 2*pi + y(1);
    }

    y(2) -= (x_(0)*x_(2) + x_(1)*x_(3)) / sqrt(x_(0)*x_(0) + x_(1)*x_(1));

    //cout<<"Ang: "<<" "<<z(1)<<" "<<atan2(x_(1), x_(0))<<" "<<y(1)<<endl;
    //cout<<"Ro: "<<" "<<z(0)<<" "<<sqrt(x_(0)*x_(0) + x_(1)*x_(1))<<endl;

    MatrixXd Ht = Hj.transpose();

    MatrixXd R = MatrixXd(3, 3);
    R << 0.0925, 0, 0,
         0, 0.0925, 0,
         0, 0, 0.0925;

    MatrixXd S = Hj * P_ * Ht + R;

    MatrixXd Si = S.inverse();
    MatrixXd K =  P_ * Ht * Si;

    x_ = x_ + (K * y);
    P_ = (I_ - K * Hj) * P_;
}
