CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I.
TARGET   = GameC
GENERATOR= genGraph
TESTS    = runTests
SRCS     = main.cpp PlagueGame.cpp Player.cpp
OBJS     = $(SRCS:.cpp=.o)

all: $(TARGET) $(GENERATOR) $(TESTS)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(GENERATOR): tools/genGraph.cpp
	$(CXX) $(CXXFLAGS) -o $@ tools/genGraph.cpp

$(TESTS): tests/tests.cpp PlagueGame.o Player.o
	$(CXX) $(CXXFLAGS) -o $@ tests/tests.cpp PlagueGame.o Player.o

%.o: %.cpp PlagueGame.h Player.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TESTS)
	./$(TESTS)

clean:
	rm -f $(OBJS) $(TARGET) $(GENERATOR) $(TESTS)

.PHONY: all run test clean
