
#include "stdafx.hpp"

#include "Commons.hpp"

namespace{
//XXX 乱数ジェネレータが分散してて気持ち悪い
static const unsigned long _Seed = 20130413;
boost::mt19937 _Gen(_Seed);
boost::uniform_01<> _Dest;
}

/*!
 */
dl::CNet::CNet(std::size_t is, std::size_t os) :
		w(is, os), b(os) {
	for (int i = 0; i < b.rows(); i++) {
		b(i) = _Dest(_Gen)*2.0f - -1.0f;
	}

	for (int i = 0; i < w.rows(); i++) {
		for (int j = 0; j < w.cols(); j++) {
			w(i, j) = _Dest(_Gen)*2.0f - 1.0f;
		}
	}
}

/*!
 */
dl::CNet::CNet(const Eigen::MatrixXf& w_, const Eigen::VectorXf& b_):
w(w_), b(b_){
}
