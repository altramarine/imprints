# IDIR =include
# CC=g++
# CFLAGS=-I$(IDIR) -O3 -g -Wall -march=native
# ODIR=src
# LIBS=-lm

# _DEPS = main.h imprints.h print.h utils.h queries.h zonemaps.h
# DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# _OBJ = main.o
# OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

# $(ODIR)/%.o: %.c $(DEPS)
# 	$(CC) -c -o $@ $< $(CFLAGS)

# imprints: $(OBJ)
# 	$(CC) -o $@ $^ $(CFLAGS) $(LIBS)

# .PHONY: clean

# clean:
# 	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~
