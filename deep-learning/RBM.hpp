#pragma once

#include "Commons.hpp"

namespace dl {
/*!
 */
class CRBM {
private:
	Eigen::VectorXf _V;
	Eigen::VectorXf _H;
	Eigen::VectorXf _B;
	Eigen::VectorXf _C;
	Eigen::MatrixXf _W;

public:
	/*! コンストラクタ
	 */
	CRBM(size_t vs, size_t hs);

	/*! データセットを元に学習を一回行う
	 */
	float Learn(const std::vector<Eigen::VectorXf>& data_set);

	/*! データセットを元に学習を行う
	 * 値が収束するまで学習を繰り返す
	 */
	void Learn(const std::vector<Eigen::VectorXf>& data_set, float eps, int min_loop_num=1e3);

	/*!
	 */
	inline Eigen::VectorXf ContDiv(const Eigen::VectorXf& vi) const{
		assert(vi.rows()==VisSize());
		return dl::Sigmoid(_B+_W * dl::Sigmoid(_C+_W.transpose() * vi));
	}

	/*!
	 */
	inline Eigen::VectorXf GetHidden(const Eigen::VectorXf& vi) const{
		assert(vi.rows()==VisSize());
		return dl::Sigmoid(_C+_W.transpose()*vi);
	}

	/*!
	 */
	CNet CreateNet() const{
		return CNet(_W, _C);
	}

	/*!
	 */
	int VisSize() const{
		return _V.rows();
	}

	/*!
	 */
	int HidSize() const{
		return _H.rows();
	}
};
}
