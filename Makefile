CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99
LDFLAGS = -lm

TARGET = test_strmm
LIB = libstrmm.a

all: $(TARGET) $(LIB)

$(TARGET): test_strmm.c strmm.c strmm.h
	$(CC) $(CFLAGS) -o $(TARGET) test_strmm.c strmm.c $(LDFLAGS)

$(LIB): strmm.c strmm.h
	$(CC) $(CFLAGS) -c strmm.c -o strmm.o
	ar rcs $(LIB) strmm.o

test: $(TARGET)
	./$(TARGET)

clean:
	del *.o *.exe *.lib 2>nul || rm -f *.o *.exe *.lib

.PHONY: all test clean
