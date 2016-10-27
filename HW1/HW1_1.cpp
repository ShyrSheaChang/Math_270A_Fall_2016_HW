#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <cmath>
#include <math.h> 

using namespace Eigen;

void calc_V(const Eigen::Matrix2f& C, Eigen::Matrix2f& V) {

	Eigen::Matrix2f Sigma;
	float tau, t_2, c_2, s_2;

	tau = (C(1, 1) - C(0, 0)) / (2 * C(1, 0));
	if (tau>0) {
		t_2 = tau - sqrt(1 + tau*tau);
	}
	else {
		t_2 = tau + sqrt(1 + tau*tau);
	}
	c_2 = 1 / sqrt(1 + t_2*t_2);
	s_2 = t_2*c_2;
	V << c_2, -s_2, s_2, c_2;
}


void signconvention(Matrix2f& U, Matrix2f& V, Vector2f& sigma) {
	Vector2f tempvec;
	float tempf;
	if (sigma(0)<0 && sigma(1)<0) {
		U = -U;
		sigma = -sigma;
	}
	if (sigma(0)<0) {
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
	else if (sigma(0) >= 0 && sigma(1) >= 0) {
		if (sigma(1) > sigma(0)) {
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

void My_SVD(const Eigen::Matrix2f& F, Eigen::Matrix2f& U, Eigen::Matrix2f& sigma, Eigen::Matrix2f& V)
{
	Eigen::Matrix2f C,A,Sigma;
	Eigen::Vector2f b,v,sigma_vec;
	Eigen::JacobiRotation<float> G;
	float a, c_U, s_U;

	

	C = F.transpose()*F;
	calc_V(C,V);
	
	A = F*V;
	
	G.makeGivens(A(0,0),A(1,0));
	v<<1,0;
    v.applyOnTheLeft(0,1,G);
    c_U = v(0);
    s_U = -v(1);
    U<<c_U,s_U,-s_U,c_U;
    Sigma = A;
    Sigma.applyOnTheLeft(0,1,G.adjoint());
    //std::cout<<Sigma;
	sigma_vec(0) = Sigma(0,0);
	sigma_vec(1) = Sigma(1,1);

	signconvention(U, V, sigma_vec);

	sigma << sigma_vec(0), 0,
		0, sigma_vec(1);

	
	
}

int main() {

	Eigen::Matrix2f F, U, sigma, V;


	F << 10, 2,
		3, 4;

	My_SVD(F, U, sigma, V);

	



	std::cout << U << std::endl << std::endl;

	std::cout << U*U.transpose() << std::endl << std::endl;

	std::cout << V << std::endl << std::endl;

	std::cout << V*V.transpose() << std::endl << std::endl;

	std::cout << sigma << std::endl << std::endl;

	std::cout << U*sigma*V.adjoint() << std::endl << std::endl;

	std::cout << F << std::endl;

	system("pause");

	return 0;
}
