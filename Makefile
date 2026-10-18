CC      ?= gcc
CFLAGS  ?= -O3 -Wall -Wextra -fopenmp
LDFLAGS ?= -fopenmp
LDLIBS  ?= -lavutil

BINDIR  ?= .
TARGETS  = $(BINDIR)/scan_mp3 $(BINDIR)/repair_mp3

.PHONY: all clean

all: $(TARGETS)

$(BINDIR)/scan_mp3: src/scan_mp3.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(BINDIR)/repair_mp3: src/repair_mp3.c
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGETS)
