#!gnumake

INCLUDE = -I. -I../../$(BMS_OSNAME)/include -I$(HALLD_RECON_HOME)/$(BMS_OSNAME)/include -I/group/halld/Software/builds/$(BMS_OSNAME)/gluex_root_analysis/gluex_root_analysis-$(GLUEX_ROOT_ANALYSIS_VERSION)/$(BMS_OSNAME)/include/DSelector/
ROOTFLAGS = $(shell $(ROOTSYS)/bin/root-config --cflags) 
CXXFLAGS = -O -Wall -fPIC 
FLAGS = $(CXXFLAGS) $(ROOTFLAGS) $(INCLUDE)
ROOTGLIBS = $(shell $(ROOTSYS)/bin/root-config --glibs) 
LIBS = $(ROOTGLIBS) -L/group/halld/Software/builds/$(BMS_OSNAME)/gluex_root_analysis/gluex_root_analysis-$(GLUEX_ROOT_ANALYSIS_VERSION)/$(BMS_OSNAME)/lib/ -lDSelector

all: install

install: build_it install_it

build_it: mkdir_build ./bin/genAnalysis

mkdir_build: 
	@mkdir -p ./bin

install_it: 
	@mkdir -p ./bin 
	@cp ./bin/genAnalysis ./bin/

./bin/genAnalysis: genAnalysis.cc 
	@rm -f ./bin/genAnalysis 
	g++ $(FLAGS) genAnalysis.cc -o ./bin/genAnalysis $(LIBS) 
	@chmod 755 ./bin/genAnalysis

clean: 
	@rm -rf ./bin
	@rm -rf ./bin/genAnalysis

env: 
	@echo C++ $(C++)
	@echo BMS_OSNAME $(BMS_OSNAME)
	@echo INCLUDE $(INCLUDE) 
	@echo FLAGS $(FLAGS)
	@echo LIBS $(LIBS)

