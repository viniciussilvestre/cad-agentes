CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -Ibase
TARGET   = GameC
GENERATOR= genGraph
TESTS    = runTests
SRCS     = base/main.cpp base/game.cpp base/player.cpp
OBJS     = $(SRCS:.cpp=.o)
HEADERS  = base/game.h base/player.h base/enums.h base/constants.h

all: $(TARGET) $(GENERATOR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(GENERATOR): tools/genGraph.cpp
	$(CXX) $(CXXFLAGS) -o $@ tools/genGraph.cpp

$(TESTS): tests/tests.cpp base/game.cpp base/player.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ tests/tests.cpp base/game.cpp base/player.cpp

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TESTS)
	./$(TESTS)

clean:
	rm -f $(OBJS) $(TARGET) $(GENERATOR) $(TESTS)

.PHONY: all run test clean
