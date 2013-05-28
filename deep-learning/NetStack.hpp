#pragma once

#include "Commons.hpp"

namespace dl {
/*!
 *
 */
class CNetStack {
public:
	typedef std::vector<CNet>::const_iterator const_iterator;

private:
	std::vector<CNet> _Stack;

public:
	/*!
	 */
	CNetStack() {
	}

	/*!
	 */
	void Push(const CNet& net) {
		_Stack.push_back(net);
	}

	/*!
	 */
	Eigen::VectorXf GetOutput(const Eigen::VectorXf& vi) {
		assert(vi.rows()==_Stack.front().InSize());
		CheckConsistency();
		Eigen::VectorXf vo = vi;
		BOOST_FOREACH(const CNet& i, _Stack) {
			vo = Sigmoid(i.b+i.w.transpose()*vo);
		}
		return vo;
	}

	/*!
	 */
	int InSize() {
		return _Stack.begin()->InSize();
	}

	/*!
	 */
	int OutSize() {
		return _Stack.back().OutSize();
	}

	/*!
	 */
	const_iterator begin() const {
		return _Stack.begin();
	}

	/*!
	 */
	const_iterator end() const {
		return _Stack.end();
	}

	/*!
	 */
	inline void CheckConsistency() {
		for (int i = 0; i < _Stack.size() - 1; i++) {
			assert(_Stack[i].OutSize() == _Stack[i + 1].InSize());
		}
	}
};
}
