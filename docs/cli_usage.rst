CLI Usage
=========

We provide a command-line interface for running the full pipeline, with a command for
each step. Each sub-command can take a YAML file for settings (some settings are
passed directly to the command line). See the `Full Analysis Example Tutorial <demos/full-analysis>`_
for a full workflow of analysing real data.

Here, we give a basic overview of the commands available, and their respective options.
Note that ``--help`` can be used at any time on the command line for any command.

Commands
--------

.. click:: edges_analysis.cli:process
   :prog: process
   :nested: full
