#include "stdafx.hpp"

#include "NeuralNet.hpp"

using namespace std;
using namespace Eigen;

namespace {
/*! シグモイド関数
 */
inline double _Sigmoid(double x) {
	return 1 / (1 + std::exp(-x));
}

inline VectorXd _Sigmoid(const VectorXd& x) {
	VectorXd o(x.rows(), 1);
	for (int i = 0; i < x.rows(); i++) {
		o(i) = _Sigmoid(x(i));
	}
	return o;
}

/*! 指定層の誤差(delta)を得る
 */
VectorXd _Delta(const VectorXd& x, const VectorXd& e, const nn::CNet& n) {
	VectorXd dx = n.w * e;
	for (size_t i = 0; i < dx.rows(); i++) {
		dx(i) = x(i) * (1 - x(i)) * dx(i);
	}
	return dx;
}

class _CRandomGenerator {
private:
	boost::mt19937 _Generator;
	boost::uniform_01<> _Destination;
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > _Random;
	static const unsigned long _Seed = 20130413;

public:
	/*!
	 */
	_CRandomGenerator() :
			_Generator(_Seed), _Destination(), _Random(_Generator, _Destination) {
	}

	/*!
	 */
	double operator ()() {
		return _Random();
	}
} _R;
}

nn::CNet::CNet(std::size_t is, std::size_t os) :
		w(is, os), b(os) {
	for (int i = 0; i < b.rows(); i++) {
		b(i) = _R() - 0.5;
	}

	for (int i = 0; i < w.rows(); i++) {
		for (int j = 0; j < w.cols(); j++) {
			w(i, j) = _R() - 0.5;
		}
	}
}

/*! コンストラクタ.
 */
nn::CNeuralNet::CNeuralNet(double eta, std::size_t size1, std::size_t size2) :
		_Eta(eta) {
	_N.push_back(CNet(size1, size2));
	_L.push_back(CLayer(size1));
	_L.push_back(CLayer(size2));
}

/*!
 */
nn::CNeuralNet::~CNeuralNet() {
}

/*! 適当な初期値で層を追加
 */
void nn::CNeuralNet::AddLayer(std::size_t size) {
	_N.push_back(CNet(_N.back().b.rows(), size));
	_L.push_back(CLayer(size));
}

/*! 入力信号から出力信号を得る
 */
Eigen::VectorXd nn::CNeuralNet::GetOutput(Eigen::VectorXd vi) {
	//入力層に値を設定
	_L.begin()->z = vi;
	//各層の出力値を計算
	for (int i = 0; i < _N.size(); i++) {
		CLayer& x = _L.at(i);
		CLayer& y = _L.at(i + 1);
		CNet& n = _N.at(i);
		y.a = n.b + n.w.transpose() * x.z;
		y.z = _Sigmoid(y.a);
	}
	return _L.back().a;
}

/*! 入力信号と教師信号から学習を行う
 */
void nn::CNeuralNet::Learn(const Eigen::VectorXd& vi,
		const Eigen::VectorXd& vt) {
	//各層の出力値を計算
	GetOutput(vi);
	//各層の誤差を計算
	_L.back().d = _L.back().a - vt;
	for (int i = _N.size() - 1; 0 <= i; i--) {
		_L[i].d = _Delta(_L[i].z, _L[i + 1].d, _N[i]);
	}
	//重みとバイアス項目をすべて更新
	for (size_t i = 0; i < _N.size(); i++) {
		CNet& n = _N[i];
		//誤差を元に重みとバイアスを更新
		n.w = n.w - _Eta * _L[i].z * _L[i + 1].d.transpose();
		n.b = n.b - _Eta * _L[i + 1].d;
		//重みとバイアスに正則化をかける
//		n.w = (1.0 - 0.01) * n.w;
//		n.b = (1.0 - 0.01) * n.b;
	}
}
