FLAGS = -std=c11 -Wall -Wextra -pedantic

LIBRARIES = -lm

OUT_FILE = out/ml

build: ml.c
	gcc $(FLAGS) ml.c $(LIBRARIES) -o $(OUT_FILE)
	./out/ml

debug: ml.c
	gcc $(FLAGS) -ggdb ml.c $(LIBRARIES) -o $(OUT_FILE)
	gdb ./out/ml
