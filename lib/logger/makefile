CC = g++
DEBUG = -g #-O3
CFLAGS = -std=c++11
CPPUNITFLAGS = -lcppunit

.PHONY : test clean count

build/%.o: src/%.cpp src/%.h
	@echo -n "Compiling $@..."	
	@$(CC) -c -o $@ $< -w $(CFLAGS) $(DEBUG)
	@echo " Done."

test/build/testLog: test/TestLog.cpp build/Log.o
	@echo -n "Compiling $@..."
	@$(CC) -o $@ $^ $(CFLAGS) $(DEBUG) $(CPPUNITFLAGS)
	@echo " Done."

test: test/build/testLog
	@./test/build/testLog

clean:
	@echo -n "Cleaning..."
	@rm -f build/*
	@rm -f test/build/*
	@rm -f test/results/*
	@rm -f log/*	
	@echo " Done."

