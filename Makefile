CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I. -Iincompleto
TARGET   = GameC
GENERATOR= genGraph
TESTS    = runTests
SRCS     = incompleto/main.cpp incompleto/game.cpp incompleto/player.cpp
OBJS     = $(SRCS:.cpp=.o)
HEADERS  = incompleto/game.h incompleto/player.h incompleto/enums.h incompleto/constants.h

all: $(TARGET) $(GENERATOR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(GENERATOR): tools/genGraph.cpp
	$(CXX) $(CXXFLAGS) -o $@ tools/genGraph.cpp

$(TESTS): tests/tests.cpp incompleto/game.cpp incompleto/player.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ tests/tests.cpp incompleto/game.cpp incompleto/player.cpp

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

test: $(TESTS)
	./$(TESTS)

clean:
	rm -f $(OBJS) $(TARGET) $(GENERATOR) $(TESTS)

.PHONY: all run test clean
