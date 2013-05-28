#pragma once

#include "Commons.hpp"
#include "NetStack.hpp"

namespace dl {
/*! ニューラルネットクラス
 */
class CNeuralNet {
private:
	std::vector<CNet> _N;
	std::vector<CLayer> _L;
	double _Eta;

public:
	/*! コンストラクタ.
	 */
	CNeuralNet(double eta, const CNetStack& stack);

	/*!
	 */
	~CNeuralNet();

	/*! 入力信号から出力信号を得る
	 */
	inline const Eigen::VectorXf& GetOutput(Eigen::VectorXf vi){
		_CalcOutput(vi);
		return _L.back().z;
	}

	/*! 入力信号と教師信号から逐次学習を行う
	 */
	void SeqLearn(const PairType& pair);

	/*! データセットからバッチ学習を行う
	 */
	void BatchLearn(const std::vector<PairType>& data_set);

	/*!
	*/
	void Display();

	/*!
	 */
	CNet CreateNet(){
		return _N.back();
	}
private:
	/*! 与えられた入力から各層の出力値を計算
	*/
	void _CalcOutput(const Eigen::VectorXf& vi);

	/*! 与えられたペアから誤差を計算
	*/
	void _CalcError(const Eigen::VectorXf& vt);
};
}
