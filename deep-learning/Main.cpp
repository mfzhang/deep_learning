#include "stdafx.hpp"

#include "NeuralNet.hpp"
#include "RBM.hpp"
#include "MonoImage.hpp"
#include "NetStack.hpp"

#include "linux-wrapper/GNUPlotWrapper.hpp"

using namespace std;
using namespace Eigen;

namespace {
const float _Eps = 1.0e-6;				//収束条件
const float _Eta = 0.2;					//学習率
const int _ReduceStep = 3;				//１階層ごとのニューロン減少数
const int _ImgLen = 8;					//画像の一片のながさ
const int _ImgSize = _ImgLen * _ImgLen;	//画像ベクトルの次元数
const int _MidDepth = 16;				//階層の深さ(+1の深さのニューラルネットが完成する)
boost::mt19937 _Gen(20130413);
}

inline float UniformSin(float x) {
	//return 0.25f*sin(M_PI*x)+0.5f;
	//return 0.25*sin(4*M_PI*x)+0.5f;
	return 0.5*((x* x) + 0.1 * sin(10 * M_PI * x))+0.25f;
	//return 0.25f*x+0.5f;
	//return x;
}

void nn_func_test();
void nn_test();
void rbm_test();
void dbn_test();

int main() {
	//関数近似テスト
	nn_func_test();
	return 0;
	//nn_test();
	{
		boost::timer Timer;
		nn_test();
		cerr << "total : " << Timer.elapsed() << "[sec]" << endl;
	}
	//rbm_test();
	{
		boost::timer Timer;
		//dbn_test();
		cerr << "total : " << Timer.elapsed() << "[sec]" << endl;
	}
}

/*! 関数近似のテスト
 */
void nn_func_test() {
	//データセットを生成
	vector<dl::PairType> DataSet;
	DataSet.reserve(200);
	dl::PairType Pair;
	Pair.first = VectorXf(1);
	Pair.second = VectorXf(1);
	for (int i = 0; i < 200; ++i) {
		float x = static_cast<float>(i) / 100.0f - 1.0f;
		Pair.first(0) = x;
		Pair.second(0) = UniformSin(x);
		DataSet.push_back(Pair);
	}

	//ニューラルネットを生成
	dl::CNetStack Stack;
	Stack.Push(dl::CNet(1, 8));
	//Stack.Push(dl::CNet(48, 48));
	Stack.Push(dl::CNet(8, 1));
	dl::CNeuralNet NN(_Eta, Stack);
	float LastError = numeric_limits<float>::max();
	//NN.BatchLearn(DataSet, _Eps);
	NN.BatchLearniRPROPminus(DataSet, 1e-4, 1e3);
//	boost::uniform_int<> Dist(0, DataSet.size()-1);
//	for(int i=0; i<1e7; ++i){
//		NN.SeqLearn(DataSet[Dist(_Gen)]);
//	}

	//プロットを標準出力に投げる
	utl::lnx::CPlotter Plotter;
	BOOST_FOREACH(const dl::PairType& i, DataSet) {
		VectorXf vo = NN.GetOutput(i.first);
		Plotter.Push("training", i.first(0), i.second(0));
		Plotter.Push("learned", i.first(0), vo(0));
	}

	Plotter.Write("result.png");
}

/*!
 */
void nn_test() {
	cerr << "start nn test" << endl;
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
	NN.BatchLearn(DataSet, _Eps);

	// 結果を吐き出す
	cerr << "test nn" << endl;
	int Ok = 0;
	int Bad = 0;
	int Fuzzy=0;
	BOOST_FOREACH(const dl::PairType& i, DataSet) {
		VectorXf vo;
		vo = NN.GetOutput(i.first);
		cerr << vo(0) << ", ";
		if( 1.0f*1.0f/3.0f<vo(0) && vo(0)<1.0f*2.0f/3.0f ){
			Fuzzy++;
			continue;
		}
		if (0.5f < vo(0) && 0.5f < i.second(0)) {
			Ok++;
		} else if (vo(0) < 0.5f && i.second(0) < 0.5f) {
			Ok++;
		} else {
			Bad++;
		}
	}
	cerr << "ok : " << Ok << endl;
	cerr << "bad : " << Bad << endl;
	cerr << "fuzzy : " << Fuzzy << endl;
}

void rbm_test() {
}

void dbn_test() {
	boost::timer Timer;
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
	cerr << "time : " << Timer.elapsed() << "[sec]" << endl;
	//BRMのスタックを生成
	dl::CNetStack NetStack;
	for (int i = 0; i < _MidDepth; i++) {
		cerr << "start : " << i << endl;
		//学習するRBMを生成
		dl::CRBM RBM(_ImgSize - i * _ReduceStep,
				_ImgSize - (i + 1) * _ReduceStep);
		//データセットを元に学習を行う
		RBM.Learn(InputSet, _Eps);
		//学習済みRBMをスタックに積む
		NetStack.Push(RBM.CreateNet());
		//データセットを更新
		BOOST_FOREACH(VectorXf& j, InputSet) {
			j = RBM.GetHidden(j);
		}
		cerr << "end " << Timer.elapsed() << endl;
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
	OutNN.BatchLearn(DataSet, _Eps);

	//学習済みパーセプトロンをスタックに積む
	NetStack.Push(OutNN.CreateNet());

	cerr << "ok : " << Timer.elapsed() << "[sec]" << endl;

	cerr << "fine tune" << endl;

	//トータルで学習
	dl::CNeuralNet TotalNN(_Eta, NetStack);
	TotalNN.BatchLearn(DataSet, _Eps);

	cerr << "ok : " << Timer.elapsed() << "[sec]" << endl;

	//学習結果をテスト
	// 結果を吐き出す
	int Ok = 0;
	int Bad = 0;
	int Fuzzy=0;
	BOOST_FOREACH(const dl::PairType& i, DataSet) {
		VectorXf vo;
		vo = TotalNN.GetOutput(i.first);
		//cout << vo(0) << ", ";
		if( 1.0f*1.0f/3.0f<vo(0) && vo(0)<1.0f*2.0f/3.0f ){
			Fuzzy++;
			continue;
		}
		if (0.5f < vo(0) && 0.5f < i.second(0)) {
			Ok++;
		} else if (vo(0) < 0.5f && i.second(0) < 0.5f) {
			Ok++;
		} else {
			Bad++;
		}
	}
	cerr << "ok : " << Ok << endl;
	cerr << "bad : " << Bad << endl;
	cerr << "fuzzy : " << Fuzzy << endl;
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

