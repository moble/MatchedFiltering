Quick start
===========

  * The presentation can be found
    [here](http://moble.github.io/MatchedFiltering/Presentation.slides.html).
  * A preview of the notebook can be seen
    [here](http://nbviewer.ipython.org/github/moble/MatchedFiltering/blob/gh-pages/MatchedFiltering.ipynb),
    but note that there are cool interactive things that are missing
    unless you download and run it yourself.


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

The material is presented as a Jupyter notebook — which looks and acts basically like Mathematica.
The notebook includes text explaining the concepts and code.  This allows the explanations to be
included (with latex equations) right among the code, and all in a live python session.  No
familiarity with python is necessary for the student, though the computer will need to be set up by
someone with good technical skills.

A second notebook is also included for more demonstrations of the Fourier transform.  This notebook
makes it easy to record a sound — tuning forks, musical instruments, or whatever the student is
curious about — and look at its Fourier transform.  This encourages the student to play with the
ideas a little, experimenting to gain understanding.



To run the code
===============

If you are reading this README from the github project page, you should first download the package
to your own computer with something like
```bash
git clone --depth 1 https://github.com/moble/MatchedFiltering.git
```
Otherwise, you are presumably reading this in the code itself.

You also need `ipython`, `matplotlib`, `numpy`, `scipy`, `h5py`, `jupyter`, a very recent (>5.0)
version of `ipywidgets`, and `widgetsnbextension` installed.  The easiest way to do all this is to
just [install `conda`](https://www.continuum.io/downloads) and run

```bash
conda update -c conda-forge -y --all
conda install -c conda-forge ipython jupyter notebook scipy matplotlib ipywidgets widgetsnbextension
```
Otherwise, all of these packages can also be installed using `pip`.

Now, from the `MatchedFiltering` directory, issue the command
```bash
    jupyter notebook MatchedFiltering.ipynb
```
This will start an ipython session, but should switch to your default web browser, where you will
interact with the session.



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
caveat here is that the students will need to install the dependencies.
With [anaconda](https://store.continuum.io/anaconda/), this is not a problem.  Assuming the students
can run it, there are questions included in the notebook.  Their answers could be turned in as a
homework assignment, or a quiz given on the material to ensure that students actually go through the
notebook.

If this will be presented together in a computer lab, it is best if things are set up as much as
possible on each computer beforehand.  The computers need to be using different accounts (with home
directories not on a shared file system), or ipython will get screwed up and run into errors.
