#pragma once

namespace nn {
/*! 結合強度クラス
 */
struct CNet {
	Eigen::MatrixXd w;	//結合強度
	Eigen::VectorXd b;	//

	CNet(std::size_t is, std::size_t os);
};

/*! 層クラス
 */
struct CLayer {
	Eigen::VectorXd a;	//ニューロン入力値
	Eigen::VectorXd z;	//ニューロン出力値
	Eigen::VectorXd d;	//誤差値

	CLayer(std::size_t size):
	a(size, 1), z(size, 1), d(size, 1){
	}
};

/*! ニューラルネットクラス
 */
class CNeuralNet {
private:
	std::vector<CNet> _N;
	std::vector<CLayer> _L;
	double _Eta;

public:
	/*! デフォルトコンストラクタ.
	 */
	CNeuralNet(double eta, std::size_t size1, std::size_t size2);

	/*!
	 */
	~CNeuralNet();

	/*! 適当な初期値で層を追加
	 */
	void AddLayer(std::size_t size);

	/*! 入力信号から出力信号を得る
	 */
	Eigen::VectorXd GetOutput(Eigen::VectorXd vi);

	/*! 入力信号と教師信号から学習を行う
	 */
	void Learn(const Eigen::VectorXd& vi, const Eigen::VectorXd& vt);
};
}
