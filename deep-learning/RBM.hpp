#pragma once

namespace rbm {
/*!
 *
 */
class CRBM {
private:
	Eigen::VectorXd v;
	Eigen::VectorXd h;
	Eigen::VectorXd b;
	Eigen::VectorXd c;
	Eigen::MatrixXd w;

public:
	/*! コンストラクタ
	 */
	CRBM(size_t vs, size_t hs) :
			v(vs, 1), h(hs, 1), b(vs, 1), c(hs, 1), w(vs, hs) {
	}

	/*!
	 */
	void Update(const Eigen::VectorXd& iv);
};
}
