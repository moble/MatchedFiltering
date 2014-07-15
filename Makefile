ipynbs := $(wildcard *.ipynb)

all: ${ipynbs:.ipynb=.slides.html}
	@:

.PHONY: all

%.slides.html: %.ipynb
	$(shell ipython nbconvert $< --to slides --template output_toggle.tpl)
