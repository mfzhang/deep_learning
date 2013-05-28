
#include "stdafx.hpp"

#include "MonoImage.hpp"

using namespace std;
using namespace Eigen;

namespace{
//XXX 乱数ジェネレータが分散してて気持ち悪い
static const unsigned long _Seed = 20130413;
boost::mt19937 _Gen(_Seed);
}

/*!
*/
mi::CMonoImage::CMonoImage(std::size_t width, std::size_t height)
:_Width(width), _Height(height), _Image(VectorXf::Zero(width*height, 1)){
}

/*!
*/
void mi::CMonoImage::Display(){
	for(int y=0; y<_Height; y++){
		for(int x=0; x<_Width; x++){
			if( 0.5 <(*this)(x, y) ){
				cout << "##";
			}else{
				cout << "--";
			}
		}
		cout << "\n";
	}
	cout << flush;
}

/*! 画像に線分を描画
 */
namespace{
void PutLine(mi::CMonoImage& img, int sx, int sy, int ex, int ey){
	if( abs(ey-sy)<abs(ex-sx) ){
		if( ex<sx ){
			std::swap(sx, ex);
			swap(sy, ey);
		}
		int w = ex-sx;
		int h = ey-sy;
		for(int i=0; i<=w; i++){
			img(sx+i, sy+(h*i/w)) = 1.0f;
		}
	}else{
		if( ey<sy ){
			swap(sx, ex);
			swap(sy, ey);
		}
		int w = ex-sx;
		int h = ey-sy;
		for(int i=0; i<=h; i++){
			img(sx+(w*i/h), sy+i) = 1.0f;
		}
	}
}
}

/*! 四角形の書かれた画像を生成
 */
mi::CMonoImage mi::CreateRect(size_t size){
	mi::CMonoImage Image(size, size);
	boost::uniform_int<> LTDist(0, size/4);
	boost::uniform_int<> RBDist(size*3/4, size-1);
	size_t ltx = LTDist(_Gen);
	size_t lty = LTDist(_Gen);
	size_t rtx = RBDist(_Gen);
	size_t rty = LTDist(_Gen);
	size_t lbx = LTDist(_Gen);
	size_t lby = RBDist(_Gen);
	size_t rbx = RBDist(_Gen);
	size_t rby = RBDist(_Gen);
	PutLine(Image, ltx, lty, rtx, rty);
	PutLine(Image, rtx, rty, rbx, rby);
	PutLine(Image, rbx, rby, lbx, lby);
	PutLine(Image, lbx, lby, ltx, lty);
	return Image;
}

/*! バツ印の書かれた画像を生成
 */
mi::CMonoImage mi::CreateCross(size_t size){
	mi::CMonoImage Image(size, size);
	boost::uniform_int<> LTDist(0, size/4);
	boost::uniform_int<> RBDist(size*3/4, size-1);
	size_t l = LTDist(_Gen);
	size_t t = LTDist(_Gen);
	size_t r = RBDist(_Gen);
	size_t b = RBDist(_Gen);
	PutLine(Image, l, t, r, b);
	PutLine(Image, r, t, l, b);
	return Image;
}
