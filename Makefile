CXX      = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I.
TARGET   = GameC
GENERATOR= genGraph
TESTS    = runTests
SRCS     = main.cpp game.cpp player.cpp
OBJS     = $(SRCS:.cpp=.o)

all: $(TARGET) $(GENERATOR)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

$(GENERATOR): tools/gen_graph.cpp
	$(CXX) $(CXXFLAGS) -o $@ tools/gen_graph.cpp


%.o: %.cpp game.h player.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(GENERATOR)

.PHONY: all run test clean
