#include "stdafx.hpp"

#include "NeuralNet.hpp"
#include "RBM.hpp"
#include "MonoImage.hpp"
#include "NetStack.hpp"

using namespace std;
using namespace Eigen;

namespace {
const float _Eta = 0.2;					//学習率
const int _ReduceStep = 2;				//１階層ごとのニューロン現象数
const int _ImgLen = 8;					//画像の一片のながさ
const int _ImgSize = _ImgLen * _ImgLen;	//画像ベクトルの次元数
const int _TryNum = 1e3;				//学習の試行回数
const int _MidDepth = 16;				//階層の深さ(+1の深さのニューラルネットが完成する)
boost::mt19937 _Gen(20130413);
}

inline float UniformSin(float x) {
	//return x * x + 0.1 * sin(10 * M_PI * x);
	return x * 0.5+0.5;
}

void nn_func_test();
void perc_test();
void p_test();
void nn_test();
void rbm_test();
void dbn_test();

int main() {
	//nn_func_test();
	nn_test();
	//rbm_test();
	//dbn_test();
}

void nn_func_test() {
	//データセットを生成
	vector<dl::PairType> DataSet;
	DataSet.reserve(1000);
	dl::PairType Pair;
	Pair.first = VectorXf(1);
	Pair.second = VectorXf(1);
	for (int i = 0; i < 1000; ++i) {
		float x = static_cast<float>(i) / 1000.0f;
		Pair.first(0) = x;
		Pair.second(0) = UniformSin(x);
		DataSet.push_back(Pair);
	}

	//ニューラルネットを生成
	dl::CNetStack Stack;
	Stack.Push(dl::CNet(1, 1));
	dl::CNeuralNet NN(_Eta, Stack);
	for (int i = 0; i < _TryNum; i++) {
		NN.BatchLearn(DataSet);
	}

	//プロットを標準出力に投げる
	BOOST_FOREACH(const dl::PairType& i, DataSet){
		cout << i.first(0) << " ";
		cout << UniformSin(i.first(0)) << " ";
		cout << i.second(0) << " ";
		cout << "\n";
	}
}

void nn_test() {
	//データセットを生成
	vector<dl::PairType> DataSet;
	dl::PairType Pair;
	Pair.first = VectorXf(_ImgSize);
	Pair.second = VectorXf(1);

	//四角を追加
	for (size_t i = 0; i < 500; i++) {
		mi::CMonoImage Image = mi::CreateRect(_ImgLen);
		Pair.first = Image.GetVector();
		Pair.second(0) = 0.0f;
		DataSet.push_back(Pair);
	}

	//バツを追加
	for (size_t i = 0; i < 500; i++) {
		mi::CMonoImage Image = mi::CreateCross(_ImgLen);
		Pair.first = Image.GetVector();
		Pair.second(0) = 1.0f;
		DataSet.push_back(Pair);
	}

	//学習に使用するニューラルネットを生成
	dl::CNetStack NetStack;
	for (int i = 0; i < _MidDepth; i++) {
		NetStack.Push(
				dl::CNet(_ImgSize - _ReduceStep * i,
						_ImgSize - _ReduceStep * (i + 1)));
	}
	NetStack.Push(dl::CNet(_ImgSize - _ReduceStep * _MidDepth, 1));
	NetStack.CheckConsistency();
	dl::CNeuralNet NN(_Eta, NetStack);

	//データセットを元に学習を行う
	for (int i = 0; i < _TryNum; i++) {
		NN.BatchLearn(DataSet);
	}

	// 結果を吐き出す
	int Ok = 0;
	int Bad = 0;
	BOOST_FOREACH(const dl::PairType& i, DataSet) {
		VectorXf vo;
		vo = NN.GetOutput(i.first);
		cout << vo(0) << ", ";
		if (0.5f < vo(0) && 0.5f < i.second(0)) {
			Ok++;
		} else if (vo(0) < 0.5f && i.second(0) < 0.5f) {
			Ok++;
		} else {
			Bad++;
		}
	}
	cout << "ok : " << Ok << endl;
	cout << "bad : " << Bad << endl;
}

void rbm_test() {
}

void dbn_test() {

	cerr << "create data set" << endl;

	//データセットを生成
	vector<VectorXf> InputSet;
	vector<VectorXf> TeacherSet;

	//四角を追加
	for (size_t i = 0; i < 500; i++) {
		mi::CMonoImage Image = mi::CreateRect(_ImgLen);
		InputSet.push_back(Image.GetVector());
		VectorXf t(1, 1);
		t(0) = 0.0f;
		TeacherSet.push_back(t);
	}

	//バツを追加
	for (size_t i = 0; i < 500; i++) {
		mi::CMonoImage Image = mi::CreateCross(_ImgLen);
		InputSet.push_back(Image.GetVector());
		VectorXf t(1, 1);
		t(0) = 1.0f;
		TeacherSet.push_back(t);
	}

	//入力と教師のペアのデータセットをいまのうちに生成
	vector<dl::PairType> DataSet;
	for (size_t i = 0; i < InputSet.size(); i++) {
		DataSet.push_back(std::make_pair(InputSet[i], TeacherSet[i]));
	}

	cerr << "create RBM" << endl;

	//BRMのスタックを生成
	dl::CNetStack NetStack;
	for (int i = 0; i < _MidDepth; i++) {
		cerr << ":" << i << endl;
		//学習するRBMを生成
		dl::CRBM RBM(_ImgSize - i * _ReduceStep,
				_ImgSize - (i + 1) * _ReduceStep);
		//データセットを元に学習を行う
		for (int j = 0; j < _TryNum; j++) {
			RBM.Learn(InputSet);
		}
		//学習済みRBMをスタックに積む
		NetStack.Push(RBM.CreateNet());
		//データセットを更新
		BOOST_FOREACH(VectorXf& j, InputSet) {
			j = RBM.GetHidden(j);
		}
	}

	cerr << "create perceptron" << endl;

	//出力層用データセット
	vector<dl::PairType> OutDataSet;
	OutDataSet.reserve(InputSet.size());
	for (size_t i = 0; i < InputSet.size(); i++) {
		OutDataSet.push_back(std::make_pair(InputSet[i], TeacherSet[i]));
	}

	//RBMスタックの出力を使って出力層のパーセプトロンを学習
	dl::CNetStack OutStack;
	OutStack.Push(dl::CNet(NetStack.OutSize(), 1));
	dl::CNeuralNet OutNN(_Eta, OutStack);
	for (int i = 0; i < _TryNum; i++) {
		OutNN.BatchLearn(OutDataSet);
	}

	//学習済みパーセプトロンをスタックに積む
	NetStack.Push(OutNN.CreateNet());

	cerr << "fine tune" << endl;

	//トータルで学習
	dl::CNeuralNet TotalNN(_Eta, NetStack);
	for (int i = 0; i < _TryNum; i++) {
		TotalNN.BatchLearn(DataSet);
	}

	//学習結果をテスト
	// 結果を吐き出す
	int Ok = 0;
	int Bad = 0;
	BOOST_FOREACH(const dl::PairType& i, DataSet) {
		VectorXf vo;
		vo = TotalNN.GetOutput(i.first);
		if (0.5f < vo(0) && 0.5f < i.second(0)) {
			Ok++;
		} else if (vo(0) < 0.5f && i.second(0) < 0.5f) {
			Ok++;
		} else {
			Bad++;
		}
	}
	cout << "ok : " << Ok << endl;
	cout << "bad : " << Bad << endl;
}

struct TagNN {
};

void mplnn_test() {
//	using namespace boost;
//	typedef mpl::vector<mpl::int_<1>, mpl::int_<3>, mpl::int_<4>, mpl::int_<1> > NNSeq;
//	typedef cnn::CNeuralNet<NNSeq, TagNN>::type NN;
//	NN::OutType vo;
//	NN::InType vi;
//
//	//学習
//	NN::Learn(vi, vo);
//
//	//出力
//	NN::GetOutput(vi);
}

