# Define the compiler and flags
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2 -g -fprofile-arcs -ftest-coverage # -g is added later for debugging

# Target executable
TARGET = NeuralNetwork

# Source files
SOURCES = NeuralNetwork.cpp MatrixMath.cpp

# Object files
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link the object files to create the executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS)

# Compile source files into object files
%.o: %.cpp MatrixMath.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(OBJECTS) $(TARGET)
	rm -f *.gcov *.gcda *.gcno

# Phony targets
.PHONY: all clean
