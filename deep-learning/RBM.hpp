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

	/*!
	 */
	void Learn(const std::vector<Eigen::VectorXf>& data_set);

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
