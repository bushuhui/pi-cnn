################################################################################
#Lib_ZY Top Makefile.
################################################################################
export MAKE             = make -j1
export MAKE_OS          = linux # win32 mac

export TOPDIR           = $(shell pwd)

#Where to put object files *.o
export BUILD_PATH       = $(TOPDIR)/build

#Where to put final applications
export BIN_PATH         = $(TOPDIR)/bin

#Where to put libraries
export LIBS_PATH        = $(TOPDIR)/libs
export LIB_MAKE_TYPE    = shared # static #


.PHONY: all libs
all : libs

libs:
	@echo "Compiling librarys of PIL"
	$(MAKE) -C src


clean :
	rm -r $(BUILD_PATH)/* 

clean_tmp:
	rm -r $(BUILD_PATH)/*


