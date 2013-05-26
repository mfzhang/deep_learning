#include "stdafx.hpp"

#include "NeuralNet.hpp"
#include "MplNeuralNet.hpp"

using namespace std;
using namespace Eigen;

inline double UniformSin(double x) {
	return x * x + 0.1 * sin(10 * M_PI * x);
//	return x * x + 0.1;
}

void nn_test() {
	nn::CNeuralNet Net(0.2, 1, 32);
	Net.AddLayer(30);
	Net.AddLayer(28);
	Net.AddLayer(28);
	Net.AddLayer(30);
	Net.AddLayer(32);
	Net.AddLayer(1);

	boost::mt19937 _Generator(20130413);
	boost::uniform_01<> _Destination;
	boost::variate_generator<boost::mt19937&, boost::uniform_01<> > _Random(
			_Generator, _Destination);

	for (int i = 0; i < 1e6; i++) {
		VectorXd vi(1);
		vi(0) = _Random();
		VectorXd vt(1);
		vt(0) = UniformSin(vi(0));
		Net.Learn(vi, vt);
	}

//教師信号プロット
	ofstream ofs("result.dat");
	ofs << "color" << endl;
	ofs << "0.0 1.0 0.0" << endl;
	ofs << "point" << endl;
	ofs << 100 << endl;
	for (int i = 0; i < 100; i++) {
		VectorXd vi(1);
		vi(0) = static_cast<double>(i) / 100.0;
		VectorXd vo(1, 1);
		vo(0) = UniformSin(vi(0));
		ofs << 10.0 * i << " ";
		ofs << 0.0 << " ";
		ofs << vo(0) * 1000 << endl;
	}

//学習結果プロット
	ofs << "color" << endl;
	ofs << "1.0 0.0 0.0" << endl;
	ofs << "point" << endl;
	ofs << 100 << endl;
	for (int i = 0; i < 100; i++) {
		VectorXd vi(1);
		vi(0) = static_cast<double>(i) / 100.0;
		VectorXd vo(Net.GetOutput(vi));
		ofs << 10.0 * i << " ";
		ofs << 0.0 << " ";
		ofs << vo(0) * 1000 << endl;
	}
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

int main() {
	nn_test();
}

