#include "stdafx.hpp"

#include "NeuralNet.hpp"

#include "Commons.hpp"

using namespace std;
using namespace Eigen;

namespace {
/*! 指定層の誤差(delta)を得る
 */
VectorXf _Delta(const VectorXf& x, const VectorXf& e, const dl::CNet& n) {
	VectorXf dx = n.w * e;
	for (int i = 0; i < dx.rows(); i++) {
		dx(i) = x(i) * (1 - x(i)) * dx(i);
	}
	return dx;
}
}

/*! コンストラクタ.
 */
dl::CNeuralNet::CNeuralNet(double eta, const CNetStack& stack) :
		_Eta(eta) {
	//レイヤーとネットを生成
	_L.push_back(CLayer(stack.begin()->InSize()));
	for (CNetStack::const_iterator i = stack.begin(); i != stack.end(); ++i) {
		_N.push_back(*i);
		_L.push_back(i->OutSize());
	}
}

/*!
 */
dl::CNeuralNet::~CNeuralNet() {
}

/*! 入力信号と教師信号から逐次学習を行う
 */
void dl::CNeuralNet::SeqLearn(const PairType& pair) {
	//各層の出力値を計算
	GetOutput(pair.first);
	//各層の誤差を計算
	_CalcError(pair.second);
	//重みとバイアス項目をすべて更新
	for (size_t i = 0; i < _N.size(); i++) {
		CNet& n = _N[i];
		//誤差を元に重みとバイアスを更新
		n.w -= _Eta * _L[i].z * _L[i + 1].d.transpose();
		n.b -= _Eta * _L[i + 1].d;
	}
}

/*! データセットからバッチ学習を行う
 */
void dl::CNeuralNet::BatchLearn(const std::vector<PairType>& data_set) {
	//重みとバイアスの更新量のカウンタを生成
	vector<CNet> AccNet;
	AccNet.reserve(_N.size());
	BOOST_FOREACH(const CNet& i, _N) {
		AccNet.push_back(i.CreateSameSizeZero());
	}
	//更新量の和を求める
	BOOST_FOREACH(const PairType& i, data_set) {
		_CalcOutput(i.first);
		_CalcError(i.second);
		for (int j = 0; j < _N.size(); j++) {
			AccNet[j].w += _L[j].z * _L[j + 1].d.transpose();
			AccNet[j].b += _L[j + 1].d;
		}
	}
	//更新量の平均を取る
	float InvDataSetSize = 1.0f / data_set.size();
	BOOST_FOREACH(CNet& i, AccNet) {
		i.w *= InvDataSetSize;
		i.b *= InvDataSetSize;
	}
	//重みとバイアスを更新
	for (size_t i = 0; i < _N.size(); i++) {
		_N[i].w -= AccNet[i].w;
		_N[i].b -= AccNet[i].b;
	}
}

/*!
 */
void dl::CNeuralNet::Display() {
	BOOST_FOREACH(const CNet& n, _N) {
		cout << "weight" << endl;
		cout << n.w << endl;
		cout << "bias" << endl;
		cout << n.b << endl;
	}
}

/*! 与えられた入力から各層の出力値を計算
 */
void dl::CNeuralNet::_CalcOutput(const Eigen::VectorXf& vi) {
	assert( vi.rows()==_L.begin()->Size());
	//入力層に値を設定
	_L.begin()->z = Sigmoid(vi);
	//各層の出力値を計算
	for (size_t i = 0; i < _N.size(); i++) {
		CLayer& x = _L.at(i);
		CLayer& y = _L.at(i + 1);
		CNet& n = _N.at(i);
		y.z = Sigmoid(n.b + n.w.transpose() * x.z);
	}
}

/*! 与えられた教師信号から誤差を計算
 */
void dl::CNeuralNet::_CalcError(const Eigen::VectorXf& vt) {
	assert(vt.rows()==_L.back().Size());
	//各層の誤差を計算
	_L.back().d = _L.back().z - vt;
	for (int i = _N.size() - 1; 0 <= i; i--) {
		_L[i].d = _Delta(_L[i].z, _L[i + 1].d, _N[i]);
	}
}
