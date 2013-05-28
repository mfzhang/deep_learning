#pragma once

namespace dl {
/*! 入力信号-教師信号ペア
 */
typedef std::pair<Eigen::VectorXf, Eigen::VectorXf> PairType;

/*! シグモイド関数
 */
inline float Sigmoid(double x) {
	return 1 / (1 + std::exp(-x));
}

/*! ベクトル用シグモイド関数
 */
inline Eigen::VectorXf Sigmoid(const Eigen::VectorXf& x) {
	Eigen::VectorXf o(x.rows(), 1);
	for (int i = 0; i < x.rows(); i++) {
		o(i) = Sigmoid(x(i));
	}
	return o;
}

/*! レイヤー間接続クラス
 */
struct CNet {
	Eigen::MatrixXf w;	//結合強度
	Eigen::VectorXf b;	//バイアス

	/*! 入力サイズと出力サイズを指定
	 */
	CNet(std::size_t is, std::size_t os);

	/*! 重みとバイアスを指定して生成
	 */
	CNet(const Eigen::MatrixXf& w_, const Eigen::VectorXf& b_);

	/*! 同サイズのゼロクリアされたインスタンスを生成
	 */
	CNet CreateSameSizeZero() const {
		return CNet(Eigen::MatrixXf::Zero(w.rows(), w.cols()),
				Eigen::VectorXf::Zero(b.rows()));
	}

	/*! 入力サイズを取得
	 */
	inline int InSize() const {
		return w.rows();
	}

	/*! 出力サイズを取得
	 */
	inline int OutSize() const {
		return w.cols();
	}
};

/*! レイヤークラス
 */
struct CLayer {
	Eigen::VectorXf z;	//ニューロン出力値
	Eigen::VectorXf d;	//誤差値

	/*!
	 */
	CLayer(std::size_t size) :
			z(size), d(size) {
	}

	/*!
	 */
	inline int Size() const {
		return z.rows();
	}
};
}
