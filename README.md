[![binder-badge]][binder] [![lite-badge]][lite]

[binder-badge]: https://mybinder.org/badge_logo.svg
[binder]: https://mybinder.org/v2/gh/moble/MatchedFiltering/binder?urlpath=lab/tree/content
[lite-badge]: https://jupyterlite.rtfd.io/en/latest/_static/badge.svg
[lite]: https://moble.github.io/MatchedFiltering/


Quick start
===========

There are several quick and easy options:

  * Click the ["binder" button][binder] above to open a live version of the main notebook that runs in the cloud.
    
    This may not work if the Binder service is too busy at the moment, and it will kill your session
    if you don't run any commands for a few minutes.  But if it is available, this will work in any
    modern browser.
  * Click the ["lite now" button][lite] above to open a live version of the main notebook that runs in your browser.
    
    This may not work with certain browser configurations — especially mobile browsers.
  * Run the notebook on your own computer if you have [anaconda](https://www.anaconda.com/) installed.  Just run these commands:
    ```bash
    conda env create moble/gw_matched_filtering_demo
    conda activate gw_matched_filtering_demo
    gw_matched_filtering_demo
    ```
    This will download and install all the necessary files, then open the notebook in your browser.


Details
=======

This package is designed to be a simple demonstration of the principles of matched filtering.  It
uses the analogy of LIGO as a microphone to explain the basic ideas, using a microphone attached to
the computer to study data as a function of time, noise sources, and real signals, as well as
headphones or a speaker to play back those signals and others.  Examples are given where a signal is
buried in the noise and extracted using matched filtering.  Real LIGO data and accurate
gravitational waveforms are also included with this code, and used for further examples to complete
the analogy.  The concepts introduced here can be applied far more widely in all areas of data
analysis.

Fourier transforms are introduced, starting with a simple example of a pure tone (which can be
played by the computer), and progressing to Fourier transforms of noise and gravitational-wave
signals.  The matched filter is then introduced by progressively building the formula with simple
explanations for each term.  Discussion of these concepts is interwoven with practice using them.

The material is presented as a Jupyter notebook — which is an interactive python session, and
includes text explaining the concepts and code.  This allows the explanations to be included (with
LaTeX equations) right among the code, and all in a live, interactive python session.  No
familiarity with python is necessary for the student, though the setup may require some basic
skills.  A presentation that introduces and motivates the material is in the root directory, and can
be [viewed on Binder
here](https://mybinder.org/v2/gh/moble/MatchedFiltering/binder?filepath=Presentation.ipynb).


To run the code
===============

If you are familiar with python packaging, you can probably figure out how to run this on your own.
Note that the required packages include ipython, jupyter, notebook, scipy, matplotlib, ipywidgets,
and widgetsnbextension.

It is much simpler to just use the [anaconda](https://www.anaconda.com/) python ecosystem.  Once
anaconda is installed, just run the following at the command prompt:

```bash
conda env create moble/gw_matched_filtering_demo
conda activate gw_matched_filtering_demo
gw_matched_filtering_demo
```

This will install all the requirements into a new conda environment, switch to that environment,
then download and run the notebook.


Notes for classroom use
=======================

There are three reasonable ways to deliver this demonstration to students: as a presentation,
individually on the students' personal computers, and together in a computer lab.

Most likely, the presentation option is the least useful to students.  Most students benefit
enormously from being able to interact with the notebook personally.  They will be more interested,
able to read along at their own pace, and play with the parameters.  If this is just not possible,
it would be best to go slowly and ask lots of questions of the students, possibly allowing one
student to actually run the commands while the teacher engages from off to the side.

A preferable option may be having the students download and run the code themselves.  The only
caveat here is that the students will need to install the dependencies.  With
[anaconda](https://www.anaconda.com/), this is not a problem.  Assuming the students can run it,
there are questions included in the notebook.  Their answers could be turned in as a homework
assignment, or a quiz given on the material to ensure that students actually go through the
notebook.

If this will be presented together in a computer lab, it is best if things are set up as much as
possible on each computer beforehand.  The computers need to be using different accounts (with home
directories not on a shared file system), or ipython will get screwed up and run into errors.


Cool video
==========
A student in one workshop pointed out [this TED Talk by the astronomer Wanda Díaz-Merced](https://www.ted.com/talks/wanda_diaz_merced_how_a_blind_astronomer_found_a_way_to_hear_the_stars)
who lost her sight, and now interacts with data by turning it into sound.  Though she uses very
different techniques from the ones we use for LIGO data, this is a very powerful example of
the importance of exploring data in different ways.
