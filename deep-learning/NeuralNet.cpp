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

/*! 与えられたfloat値の符号を残して他を1.0に
 */
inline float _Sign(float val){
	static const float Onef = 1.0f;
	static const uint32_t One = *reinterpret_cast<const uint32_t*>(&Onef);
	const uint32_t Ret = One|(*reinterpret_cast<uint32_t*>(&val) & 0x80000000);
	return *reinterpret_cast<const float*>(&Ret);
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
float dl::CNeuralNet::BatchLearn(const std::vector<PairType>& data_set) {
	//重みとバイアスの更新量のカウンタを生成
	vector<CNet> AccNet;
	AccNet.reserve(_N.size());
	BOOST_FOREACH(const CNet& i, _N) {
		AccNet.push_back(i.CreateSameSizeZero());
	}
	//誤差のカウンタ
	float Error=0;
	//すべてのデータに対して処理
	BOOST_FOREACH(const PairType& i, data_set) {
		//すべての層の出力と誤差を計算
		_CalcOutput(i.first);
		_CalcError(i.second);
		//誤差を加算
		Error += _L.back().d.norm();
		//更新量を加算
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
	//誤差を返却
	return Error/data_set.size();
}


/*! データセットからバッチ学習を行う
 * iRPROP-で学習する
 */
void dl::CNeuralNet::BatchLearniRPROPminus(const std::vector<PairType>& data_set, float eps, int convergence_count){
	static const float DeltaZero = 0.1f;
	static const float DeltaMax = 50.0f;
	static const float DeltaMin = 1e-6f;
	static const float EtaMinus = 0.5f;
	static const float EtaPlus = 1.2f;

	//更新量
	vector<CNet> Delta;
	Delta.reserve(_N.size());
	BOOST_FOREACH(const CNet& i, _N) {
		Delta.push_back(i.CreateSameSizeZero());
		for(int r=0; r<Delta.back().w.rows(); ++r){
			for(int c=0; c<Delta.back().w.cols(); ++c){
				Delta.back().w(r, c) = DeltaZero;
			}
		}
		for(int r=0; r<Delta.back().b.rows(); ++r){
			Delta.back().b(r) = DeltaZero;
		}
	}
	//１回前の重みとバイアスの傾斜
	vector<CNet> LastGradient;
	LastGradient.reserve(_N.size());
	BOOST_FOREACH(const CNet& i, _N) {
		LastGradient.push_back(i.CreateSameSizeZero());
	}
	//現在の重みとバイアスの更新量の傾斜
	vector<CNet> Gradient;
	Gradient.reserve(_N.size());
	BOOST_FOREACH(const CNet& i, _N) {
		Gradient.push_back(i.CreateSameSizeZero());
	}
	//前回の誤差
	float LastError = std::numeric_limits<float>::max();
	//収束後繰り返し回数
	int ConvergenceCount = 0;
	//収束するまで繰り返し
	for(int Count=0;;++Count){
		//データセットから傾斜とそのノルムを計算
		float Error=0;
		BOOST_FOREACH(const PairType& i, data_set) {
			//すべての層の出力と誤差を計算
			_CalcOutput(i.first);
			_CalcError(i.second);
			//誤差を加算
			Error += _L.back().d.norm();
			//更新量を加算
			for (int j = 0; j < _N.size(); j++) {
				Gradient[j].w += _L[j].z * _L[j + 1].d.transpose();
				Gradient[j].b += _L[j + 1].d;
			}
		}
		//更新量の平均を取る
		float InvDataSetSize = 1.0f / data_set.size();
		BOOST_FOREACH(CNet& i, Gradient) {
			i.w *= InvDataSetSize;
			i.b *= InvDataSetSize;
		}
		//収束したらそこで終了
		if(std::abs(LastError-Error)<eps){
			++ConvergenceCount;
			if(convergence_count<ConvergenceCount){
				return;
			}
		}else{
			ConvergenceCount = 0;
		}
		LastError = Error;
		//更新
		for (int j = 0; j < _N.size(); j++) {
			//重み
			for(int r=0; r<_N[j].w.rows(); ++r){
				for(int c=0; c<_N[j].w.cols(); ++c){
					if(0<LastGradient[j].w(r, c)*Gradient[j].w(r,c)){
						Delta[j].w(r, c) = std::min(Delta[j].w(r, c)*EtaPlus, DeltaMax);
						_N[j].w(r, c) += -_Sign(Gradient[j].w(r,c))*Delta[j].w(r,c);
						LastGradient[j].w(r, c) = Gradient[j].w(r, c);
					}else if(LastGradient[j].w(r, c)*Gradient[j].w(r,c)<0){
						Delta[j].w(r, c) = std::max(Delta[j].w(r, c)*EtaMinus, DeltaMin);
						LastGradient[j].w(r, c) = 0.f;
					}else{
						_N[j].w(r, c) += -_Sign(Gradient[j].w(r, c))*Delta[j].w(r, c);
						LastGradient[j].w(r, c) = Gradient[j].w(r, c);
					}
				}
			}
			//バイアス
			for(int r=0; r<_N[j].b.rows(); ++r){
				if(0<LastGradient[j].b(r)*Gradient[j].b(r)){
					Delta[j].b(r) = std::min(Delta[j].b(r)*EtaPlus, DeltaMax);
					_N[j].b(r) += -_Sign(Gradient[j].b(r))*Delta[j].b(r);
					LastGradient[j].b(r) = Gradient[j].b(r);
				}else if(LastGradient[j].b(r)*Gradient[j].b(r)<0){
					Delta[j].b(r) = std::max(Delta[j].b(r)*EtaMinus, DeltaMin);
					LastGradient[j].b(r) = 0.f;
				}else{
					_N[j].b(r) += -_Sign(Gradient[j].b(r))*Delta[j].b(r);
					LastGradient[j].b(r) = Gradient[j].b(r);
				}
			}
		}
	}

}

/*! 指定した誤差に収束するまで学習を行う
 */
void dl::CNeuralNet::BatchLearn(const std::vector<PairType>& data_set, float eps, int min_loop_num){
	float LastError = numeric_limits<float>::max();
	float TempError;
	for(int i=0; i<min_loop_num; i++){
		BatchLearn(data_set);
	}
	for(;;){
		TempError = BatchLearn(data_set);
		if(abs(LastError-TempError)<eps ){
			break;
		}
		LastError = TempError;
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
	_L.begin()->z = vi;
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
