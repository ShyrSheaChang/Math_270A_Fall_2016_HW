#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Jacobi>


using namespace Eigen;

void My_Polar(const Eigen::Matrix3f& F, Eigen::Matrix3f& R, Eigen::Matrix3f& S) {

	 
	Eigen::JacobiRotation<float> G;
	float tol = 1e-6, max_it = 1e4, S_diff[] = { 100,100,100 }, S_diff_max;
	int it = 0, i_rot = 0, i, j, i_max;


	

	R = MatrixXf::Identity(3, 3);
	S = F;

	for (i_max = 0; i_max < 3; i_max++) {
		if (i_max == 0)
			S_diff_max = S_diff[0];
		else if (S_diff[i_max] > S_diff[i_max - 1])
			S_diff_max = S_diff[i_max];
	}



	while (it<max_it && S_diff_max > tol) {


		for (i_rot = 0; i_rot < 3; i_rot++) {
			if (i_rot == 0) {
				i = 1;
				j = 2;
			}
			else if (i_rot == 1) {
				i = 0;
				j = 2;
			}
			else if (i_rot == 2) {
				i = 0;
				j = 1;
			}
			G.makeGivens(S(i, i) + S(j, j), S(i, j) - S(j, i));
			R.applyOnTheRight(i, j, G.adjoint());
			S.applyOnTheLeft(i, j, G);



		}

		it++;
		S_diff[0] = std::abs(S(1, 2) - S(2, 1));
		S_diff[1] = std::abs(S(0, 2) - S(2, 0));
		S_diff[2] = std::abs(S(0, 1) - S(1, 0));
		for (i_max = 0; i_max < 3; i_max++) {
			if (i_max == 0)
				S_diff_max = S_diff[0];
			else if (S_diff[i_max] > S_diff[i_max - 1])
				S_diff_max = S_diff[i_max];
		}



	}

}

int main()
{
	Eigen::Matrix3f F, R, S;

	F << 1, 2, 6,
		4, 3, 2,
		8, 4, 6;

	My_Polar(F, R, S);
	


	std::cout << R << '\n' << '\n';
	std::cout << R*R.transpose() << '\n' << '\n';
	std::cout << S << '\n' << '\n';
	std::cout << R*S << '\n' << '\n';
	std::cout << F << '\n';

	
	

	system("pause");
    
	
	return 0;

	
}
