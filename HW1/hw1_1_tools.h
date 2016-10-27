#include <iostream>
#include <Eigen/Dense>
#include <cmath>
#include <math.h> 

using namespace Eigen;



void calc_V(const Eigen::Matrix2f& C, Eigen::Matrix2f& V){

	Eigen::Matrix2f Sigma;
	float tau, t_2, c_2, s_2;

	tau = (C(1,1)-C(0,0))/(2*C(1,0));
	if (tau>0){
		t_2 = tau - sqrt(1+tau*tau);
	}
	else{
		t_2 = tau + sqrt(1+tau*tau);
	}
	c_2 = 1/sqrt(1+t_2*t_2);
	s_2 = t_2*c_2;
	V << c_2,-s_2,s_2,c_2;	
}


void signconvention(Matrix2f& U,Matrix2f& V,Vector2f& sigma){
	Vector2f tempvec;
	float tempf;
   if(sigma(0)<0 && sigma(1)<0){
   U = -U;
   sigma = -sigma;
   }
   if(sigma(0)<0){
   	tempvec = U.col(0);
   	U.col(0) << U.col(1);
   	U.col(1) = tempvec;
   	tempf = sigma(0);
   	sigma(0) = sigma(1);
   	sigma(1) = tempf;
   	tempvec = V.col(0);
   	V.col(0) << V.col(1);
   	V.col(1) = tempvec;
   	
   }
   else if(sigma(0)>=0 && sigma(1) >= 0){
   	if(sigma(1) > sigma(0)){
		tempvec = U.col(0);
		U.col(0) << U.col(1);
		U.col(1) = tempvec;
		tempf = sigma(0);
		sigma(0) = sigma(1);
		sigma(1) = tempf;
		tempvec = V.col(0);
		V.col(0) << V.col(1);
		V.col(1) = tempvec;
	   }
   }
   	
   

}


