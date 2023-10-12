COMPILER = gcc

FLAGS = -std=c11 -Wall

SOURCES = ml.c

#LIBRARIES = 

OUT_FILE = /out/ml

all :
	$(COMPILER) $(FLAGS) $(SOURCES) -o $(OUT_FILE)
