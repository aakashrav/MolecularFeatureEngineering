SOURCE=main.cpp DataGenerator.cpp
OBJECTS=$(SOURCE:.cpp=.o)
CC=g++
CFLAGS=-I./include -MP -MMD

all: tester

.cpp.o:
	$(CC) $(CFLAGS) -c $<
	@Echo successful compilation

tester: $(OBJECTS)
	$(CC) $(OBJECTS) -o $@

clean:
	rm tester $(OBJECTS)

.PHONY: all clean

