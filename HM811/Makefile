# based on https://robjhyndman.com/hyndsight/makefiles/
# compile .tex file (first runs R code to generate plots and tables if necessary) 
## using `make` in Windows cmd 
# seems to need .Rout file(s) or it will not run the corresponding R file(s). 
# in that case run `make R` first

# Usually, only these lines need changing
TEXFILE= main
RDIR= .
FIGDIR= .

# list R files
RFILES := $(wildcard $(RDIR)/*.R)
# pdf figures created by R
PDFFIGS := $(wildcard $(FIGDIR)/*.png)
# Indicator files to show R file has run
OUT_FILES:= $(RFILES:.R=.Rout)

#make the pdf given we have the outfoles
all: $(TEXFILE).pdf $(OUT_FILES)

# RUN EVERY R FILE (only if an R is newer than Rout will it need to run)
$(RDIR)/%.Rout: $(RDIR)/%.R
	R CMD BATCH $<

# Compile main tex file and show errors
$(TEXFILE).pdf: $(TEXFILE).tex $(OUT_FILES)
	latexmk -pdf -quiet $(TEXFILE)

# Run R files
R: $(OUT_FILES)

# View main tex file (for Linux, Rob used evince instead of start)
view: $(TEXFILE).pdf
	start $(TEXFILE).pdf &

# Clean up stray files
clean:
#	rm -fv $(OUT_FILES) # can't seem to delete list of files
	del /Q *.aux *.log *.toc *.blg *.bbl *.synctex.gz
	del /Q *.out *.bcf *blx.bib *.run.xml
	del /Q *.fdb_latexmk *.fls
	del /Q $(TEXFILE).pdf

.PHONY: all clean