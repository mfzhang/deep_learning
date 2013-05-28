
#include "stdafx.hpp"

#include "RBM.hpp"

#include "Commons.hpp"

using namespace std;
using namespace Eigen;

namespace{
/* 重みの更新式の片方のシグマ内を計算
 */
inline MatrixXf _CalcW(const MatrixXf& w, const VectorXf& c, const VectorXf& v){
	return v*dl::Sigmoid(c+w.transpose()*v).transpose();
}

/*! バイアスの更新式の片方のシグマ内を計算
 */
inline VectorXf _CalcB(const VectorXf& v){
	return v;
}

/*! バイアスの更新式の片方のシグマ内を計算
 */
inline VectorXf _CalcC(const MatrixXf& w, const VectorXf& c, const VectorXf& v){
	return dl::Sigmoid(c+w.transpose()*v);
}

static const unsigned long _Seed = 20130413;
boost::mt19937 _Gen(_Seed);
boost::uniform_01<> _Dest;
}

/*! コンストラクタ
 */
dl::CRBM::CRBM(size_t vs, size_t hs) :
		_V(vs, 1), _H(hs, 1), _B(vs, 1), _C(hs, 1), _W(vs, hs) {
	for(int i=0; i<_W.rows(); ++i){
		for(int j=0; j<_W.cols(); ++j){
			_W(i, j) = _Dest(_Gen)-0.5f;
		}
	}
	for(int i=0; i<_B.rows(); ++i){
		_B(i) = _Dest(_Gen)-0.5f;
	}
	for(int i=0; i<_C.rows(); ++i){
		_C(i) = _Dest(_Gen)-0.5f;
	}
}

/*!
 */
void dl::CRBM::Learn(const std::vector<Eigen::VectorXf>& data_set){
	//重みとバイアスの更新量のカウンタ
	MatrixXf SumW = MatrixXf::Zero(_W.rows(), _W.cols());
	VectorXf SumB = VectorXf::Zero(_B.rows(), 1);
	VectorXf SumC = VectorXf::Zero(_C.rows(), 1);

	//更新量を加算していく
	BOOST_FOREACH(const VectorXf& v, data_set){
		VectorXf v_hat = ContDiv(v);
		SumW += _CalcW(_W, _C, v)-_CalcW(_W, _C, v_hat);
		SumB += _CalcB(v)-_CalcB(v_hat);
		SumC += _CalcC(_W, _C, v)-_CalcC(_W, _C, v_hat);
	}

	//更新量を平均化
	float InvN=1.0f/data_set.size();
	SumW *= InvN;
	SumB *= InvN;
	SumC *= InvN;

	//重みとバイアスを更新
	_W += SumW;
	_B += SumB;
	_C += SumC;
}
