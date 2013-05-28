
#pragma once

namespace mi{
	/*!
	 */
	class CMonoImage{
	private:
		std::size_t _Width;
		std::size_t _Height;
		Eigen::VectorXf _Image;

	public:
		/*!
		*/
		CMonoImage(std::size_t width, std::size_t height);

		/*!
		*/
		float& operator ()(std::size_t width, std::size_t height){
			assert( width<_Width );
			assert( height<_Height );
			return _Image(width+height*_Width);
		}

		/*!
		*/
		float operator ()(std::size_t width, std::size_t height) const{
			assert( width<_Width );
			assert( height<_Height );
			return _Image(width+height*_Width);
		}

		/*!
		*/
		const Eigen::VectorXf& GetVector(){
			return _Image;
		}

		/*!
		*/
		void Display();
	};

	/*! 四角形の書かれた画像を生成
	 */
	mi::CMonoImage CreateRect(size_t size);

	/*! バツ印の書かれた画像を生成
	 */
	mi::CMonoImage CreateCross(size_t size);
}
