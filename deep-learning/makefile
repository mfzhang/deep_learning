#
# 私用Makefile
#
# カレントよりも下の階層のソースファイル全てを列挙してビルドする
#
# ターゲットを指定しない場合はビルドされないので
#	make all
#	make all-release
# のどちらかを設定すること
# 前者はデバッグビルドで後者はリリースビルド
#
# デバッグビルドの場合は
#	最適化なし
#	デバッグ情報付加
#	"DEBUG"マクロ定義
#
# リリースビルドの場合は
#	最適化有り(O3)
#	プロファイリング情報付加
#	"NDEBUG"マクロ定義
#
# ソースファイルの拡張子は.cppと.cuを想定
#
# 追加のライブラリのフラグは"INCLUDES"と"LIBS"に追加すること
# =で代入ではなく+=で追加していく
#
# -jオプションをつけるとプリコンパイルヘッダーの生成が終わる前にソースファイルのコンパイルが始まってしまう
# なので、makeの呼び出しで-jを付けてはいけない
# 並列ビルドは多段make内で自動的に行われるので心配はいらない
#
# プリコンパイルヘッダを使用する場合は各階層に"stdafx.hpp"を配置すること
# この"stdafx.hpp"内でインクルードされたヘッダはすべてプリコンパイルされる
# 各ソースの先頭で"stdafx.hpp"をインクルードすることで恩恵を得られる
# プリコンパイルが終わってからソースファイルのコンパイルが始まる
#
# -MMDオプションにより依存関係はすべて自動生成される
#
# CUDAと共存できる
#

# 生成されるバイナリファイル名と静的ライブラリの名前.
TARGET		= deeplearning
BIN_TARGET	= $(TARGET)
LIB_TARGET = lib$(TARGET).a

# プリコンパイルヘッダ生成用ソースをすべて列挙
PCH_HPP		= $(shell find * -name "stdafx.hpp")
PCH_GCH		= $(patsubst %.hpp,%.hpp.gch,$(PCH_HPP))

# ソースファイルから必要なファイルをすべて列挙
SRCS		= $(shell find * -name "*.cpp" -or -name "*.cu")
OBJS		= $(patsubst %.cpp,%.o,$(SRCS))
DEPENDS		= $(patsubst %.cpp,%.d,$(SRCS))

# 入れ子make情報
SUBMAKE_FLAGS		= -j --no-print-directory

# C++コンパイル情報
CXX					= g++ -MMD -MP -pipe
CXX_DEBUG_FLAGS		= -Wall -W -O0 -g -DDEBUG
CXX_RELEASE_FLAGS	= -O3 -march=native -DRELEASE -DNDEBUG

# CUDAコンパイル情報
NVCC				= nvcc -M
NVCC_DEBUG_FLAGS	= -O0 -g
NVCC_RELEASE_FLAGS	= -O2

# リンク情報
LD					= g++
LIBS				=

# ライブラリ生成情報
AR					= ar rv

# ranlib情報
RANLIB				= ranlib

# ライブラリ情報
# boostパス指定
INCLUDES	+=
LIBS		+= -lboost_program_options -lboost_system -lboost_thread -lboost_date_time -lboost_filesystem -lboost_serialization
# eigen
INCLUDES	+= $(shell pkg-config eigen3 --cflags)
LIBS		+= $(shell pkg-config eigen3 --libs)

# 未指定の場合は何もしない
.PHONY			: none
none			: ;

# clean
.PHONY			: clean
clean			:
	rm -f $(BIN_TARGET) $(LIB_TARGET)
	rm -f $(shell find * -name "*.o") $(shell find * -name "*.d") $(shell find * -name "*.hpp.gch")

# allの場合はdebugですべてビルド
.PHONY			: all
all				: SUBMAKE_FLAGS+=CXX_CONF_FLAGS="$(CXX_DEBUG_FLAGS)"
all				: SUBMAKE_FLAGS+=NVCC_CONF_FLAGS="$(NVCC_DEBUG_FLAGS)"
all				: pch
all				: binary
all				: library

# all-releaseの場合はreleaseですべてビルド
.PHONY			: all-release
all-release		: SUBMAKE_FLAGS+=CXX_CONF_FLAGS="$(CXX_RELEASE_FLAGS)"
all-release		: SUBMAKE_FLAGS+=NVCC_CONF_FLAGS="$(NVCC_RELEASE_FLAGS)"
all-release		: pch
all-release		: binary
all-release		: library

# 多重makeでプリコンパイルヘッダを生成
.PHONY			: pch
pch				:
	$(MAKE) $(SUBMAKE_FLAGS) $(PCH_GCH)

# 多重makeで実行ファイルを生成
.PHONY			: binary
binary			:
	$(MAKE) $(SUBMAKE_FLAGS) $(BIN_TARGET)

# 多重makeでライブラリを生成
.PHONY			: library
library			:
	$(MAKE) $(SUBMAKE_FLAGS) $(LIB_TARGET)

# binary
$(BIN_TARGET)	: $(OBJS)
	$(LD) -o $@ $(OBJS) $(LIBS)

# library
$(LIB_TARGET)	: $(OBJS)
	$(AR) $@ $(OBJS)
	$(RANLIB) $@

# サフィックスルール
.SUFFIXES		: .cpp .cu .o .hpp .hpp.gch

# precompile header
.hpp.hpp.gch	:
	$(CXX) -x c++-header -o $@ $(CXX_CONF_FLAGS) $(INCLUDES) $<

# cpp object
.cpp.o			:
	$(CXX) -c -o $@ $(CXX_CONF_FLAGS) $(INCLUDES) $<
	@echo "$@:$(dir $@)stdafx.hpp" >> $(patsubst %.o,%.d,$@)

# CUDA object
.cu.o			:
	$(NVCC) -c -o $@ $(NVCC_CONF_FLAGS) $<

# -MMDオプションで生成される依存関係ファイルをインクルード
-include $(DEPENDS)
