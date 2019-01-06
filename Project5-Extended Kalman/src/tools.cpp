#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  VectorXd RMSE(4);
  RMSE << 0,0,0,0;
  if (estimations.size()==0 || estimations.size()!=ground_truth.size())
  {
  	std::cout << "Invalid estimation size"<<std::endl ;
    return RMSE;
  }
  for (unsigned int i=0;i<estimations.size();++i)
  {
    VectorXd residual = estimations[i]-ground_truth[i];
    residual = residual.array()*residual.array();
    RMSE += residual;
  }
  RMSE = RMSE/estimations.size();
  RMSE = RMSE.array().sqrt();
  return RMSE;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
 MatrixXd Hj(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);
  
  float denom = px*px+py*py;
  float denom_sq = sqrt(denom);
  float denom_sq3 = denom_sq*denom_sq*denom_sq;
  if (denom<0.0001){
      std::cout << "division by 0"<<std::endl;
      return Hj;
  }
  // TODO: YOUR CODE HERE 
  
  // compute the Jacobian matrix
    Hj << px/denom_sq,py/denom_sq,0,0,
          -py/denom,px/denom,0,0,
          py*(vx*py-vy*px)/denom_sq3,px*(vy*px-vx*py)/denom_sq3,px/denom_sq,py/denom_sq;


  return Hj;
}
