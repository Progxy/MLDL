COMPILER = gcc

FLAGS = -std=c11 -Wall

SOURCES = ml.c

LIBRARIES = -lm

OUT_FILE = out/ml

all :
	$(COMPILER) $(FLAGS) $(SOURCES) $(LIBRARIES) -o $(OUT_FILE)
