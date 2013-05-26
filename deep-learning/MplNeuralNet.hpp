//#pragma once
//
//#include "stdafx.hpp"
//
//namespace cnn {
///*! 結合強度クラス
// */
//template<size_t N, size_t M, class Tag, size_t Depth>
//struct _CNet {
//	static Eigen::Matrix<float, N, M> w;	//結合強度
//	static Eigen::Matrix<float, M, 1> b;	//バイアス
//};
//
//template<size_t N, size_t M, class Tag, size_t Depth>
//Eigen::Matrix<float, N, M> _CNet<N, M, Tag, Depth>::w;	//結合強度
//
//template<size_t N, size_t M, class Tag, size_t Depth>
//Eigen::Matrix<float, M, 1> _CNet<N, M, Tag, Depth>::b;	//バイアス
//
///*! 層クラス
// */
//template<size_t N, class Tag, size_t Depth>
//struct _CLayer {
//	typedef Eigen::Matrix<float, N, 1> VectorType;
//
//	static VectorType a;	//ニューロン入力値
//	static VectorType z;	//ニューロン出力値
//	static VectorType d;	//誤差値
//};
//
//template<size_t N, class Tag, size_t Depth>
//Eigen::Matrix<float, N, 1> _CLayer<N, Tag, Depth>::a;	//ニューロン入力値
//
//template<size_t N, class Tag, size_t Depth>
//Eigen::Matrix<float, N, 1> _CLayer<N, Tag, Depth>::z;	//ニューロン出力値
//
//template<size_t N, class Tag, size_t Depth>
//Eigen::Matrix<float, N, 1> _CLayer<N, Tag, Depth>::d;	//誤差値
//
///*! ニューラルネットクラス
// */
//template<class Seq, class Tag, size_t Depth>
//class _CNeuralNet {
//private:
//	typedef typename boost::mpl::pop_front<Seq>::type _PopedSeq;
//	typedef typename boost::mpl::front<Seq>::type _N;
//	typedef typename boost::mpl::front<_PopedSeq>::type _M;
//	static const size_t _InDepth = Depth;
//	static const size_t _OutDepth = Depth - 1;
//	typedef _CNet<_N::value, _M::value, Tag, _InDepth> _Net;
//	typedef _CLayer<_N::value, Tag, _InDepth> _InLayer;
//	typedef _CLayer<_M::value, Tag, _OutDepth> _OutLayer;
//	typedef _CNeuralNet<_PopedSeq, Tag, _OutDepth> _Next;
//
//public:
//	typedef typename _Next::OutType OutType;
//	typedef typename _InLayer::VectorType InType;
//
//	static const int LayerSize = _CNeuralNet<_PopedSeq, Tag, _OutDepth>::LayerSize+1;
//
//	/*! 出力を求める
//	 */
//	static inline const OutType& GetOutput(const InType& vi){
//	}
//
//	/*! 学習を行う
//	 */
//	static inline const void Learn(const InType& vi, const OutType vt){
//		_CalcOutput(vi);
//		_CalcError(vt);
//		_RefreshNet();
//	}
//
//	/*! 入力レイヤーの指定要素を取得
//	 */
//	static inline double InLayer(int d){
//		if( d==_InDepth ){
//			return _InLayer::z(0);
//		}else{
//			return InLayer(d-1);
//		}
//	}
//
//	static inline void InLayer(int d, double val){
//		if( d==_InDepth ){
//			_InLayer::z(0) = val;
//		}else{
//			InLayer(d-1, val);
//		}
//	}
//
//private:
//	/*! 出力を取得
//	 */
//	/*! 出力を計算
//	 */
//	static inline void _CalcOutput(const InType& vi){
//		_CalcOutput(_Net::b+_Net::w.transpose()*vi);
//	}
//
//	/*! 誤差を計算
//	 */
//	static inline const InType& _CalcError(const OutType& vt){
//	}
//
//	/*! 重みを更新
//	 */
//	static inline void _RefreshNet(){
//	}
//};
//
///*! 出力層ニューラルネットクラス
// */
//template<class Seq, class Tag>
//class _CNeuralNet<Seq, Tag, 0>{
//private:
//	typedef typename boost::mpl::front<Seq>::type _N;
//	static const size_t _InDepth = 0;
//	typedef _CLayer<_N::value, Tag, _InDepth> _InLayer;
//
//public:
//	typedef typename _InLayer::VectorType OutType;
//	static const int LayerSize = 1;
//
//	/*! コンストラクタ
//	 */
//	_CNeuralNet() {
//		std::cerr << "[" << 0 << "]" << "in : " << _N::value << std::endl;
//	}
//};
//
///*! 公開用ニューラルネットクラス型
// */
//template<class Seq, class Tag>
//struct CNeuralNet{
//	typedef typename boost::mpl::size<Seq>::type _SeqSize;
//	typedef _CNeuralNet<Seq, Tag, _SeqSize::value-1> type;
//};
//}
