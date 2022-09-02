CLI Usage
---------
The intention is that full processing pipelines should be run via the CLI interface.
This interface assures that the pipeline is reproducible: it comes from a single YAML
workflow file, and each processing step that is run is necessarily a ``@gsregister``
function, so it preserves history as much as possible.

The Workflow File
~~~~~~~~~~~~~~~~~
The primary command in the CLI is ``process``. This command takes a single required
argument -- the workflow file -- and several optional arguments related to I/O.
The full option list can be found below, but here we will cover the main thing:
the workflow.

The workflow file is a YAML file that contains a list of processing steps. It should be
formatted as follows::

    globals:
       param1: value1
       param2: value2

    steps:
        - function: <<name of gsregistered function>>
          params:
            a_parameter: <<value>>
            another_parameter: <<value>>

        - function: <<name of gsregistered function>>
          name: the-first-invocation-of-foobar
          params:
            a_parameter: {{ globals.param1 }}
            another_parameter: {{ globals.param2 }}
          write: {prev_stem}.foobar.gsh5

        - ...

Note a few things. First, we have a ``globals`` section, which contains parameter values
that can be used in later steps (use these if you need to use the same value more than
once). These are interpolated into the later steps via Jinja templates, i.e. by
using double-curly braces with spaces (as seen for the second function's ``a_parameter``).
Secondly, we have a ``steps`` section, which defines a list of processing steps that will
happen in the order they are defined. Each step can have four keys:

* ``function`` is the only required key, and specifies a gsregistered function to run.
  The CLI can only find the function if it has been registered. You can get a current
  list of available functions (and their types) by using ``edges-analysis avail``.
* ``name`` gives a unique name to the step. By default, it is the function name, but
  if you use the same function more than once, you will need to specify a unique name.
* ``params`` is a dictionary of parameters to pass to the function, other than the GSData
  object itself.
* ``write`` is optional, and if included, it tells the workflow to write out a *new*
  file at this point. The value given is the filename to write. By default, if the file
  already exists, the workflow will error. Notice that the value here also uses curly
  braces. In this case, it is *not* a Jinja template, but rather a standard string-format.
  Each step has access to the variables ``prev_stem`` (i.e. the filename, without extension,
  of the last *written* step), ``prev_dir`` (i.e. the directory of the last *written* step),
  and ``fncname`` (i.e. the name of the function for this step).

.. note:: There is one "special" function that can be used that is not in the gsregister:
   the ``convert`` function. This function can be used to initially read files in a
   different format (eg. ACQ).

Using the ``process`` command
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is generally best to include a *complete* workflow in a single file -- all the way
from initial ``convert`` to the final averaging down to a single spectrum (if desired).
Keeping it in a single files means it is easier to reason about what was done to produce
the results later on (especially in conjunction with the ``history`` in the written files).
Doing this is made possible by an option to to the ``process`` command that lets you
start the workflow from any given step, which means that if the workflow fails for some
reason mid-way, you don't have to restart the whole thing (as long as you have written
checkpoints).

The typical envisaged usage of the ``process`` command, given a workflow file called
``workflow.yaml``, is::

    $ edges-analysis process workflow.yaml \
         -i "/path/to/raw/data/2015_*.acq" \
         --outdir ./my_workflow \
         --nthreads 16

Note that the input files (specified by ``-i``) can be specified with a glob, and
multiple ``-i`` options can be given. If you start with ``.acq`` files, be sure to use
``convert`` as your first step. The ``--outdir`` option tells the workflow where to
write the output data files. All filenames given in the workflow are relative to this.
It is the current directory by default.

If you only want to run a portion of the workflow, you can specify ``--stop <NAME>``,
where the name is the name of the step (or its function name) which is the last one you
want to run.

You can *resume* the workflow by simply pointing to the same output directory without
giving any inputs::

    $ edges-analysis process workflow.yaml --outdir ./my_workflow

Every time the workflow is run, a "progressfile.yaml" is written to the output
directory, containing the full specification of the run, plus some extra metadata
required to know what has already been run. You can add new input files to the workflow
by adding new ``-i`` entries::

    $ edges-analysis process workflow.yaml \
         -i "/path/to/raw/data/2016_*.acq" \
         --outdir ./my_workflow \
         --nthreads 16

This will run all the 2016 files, and then combine them with the 2015 files as necessary.
The 2015 files will not be reprocesed unless required (eg. when LST-averaging).

If you'd prefer to completely restart the process with the new files, just use the
``--restart`` option.

The ``fork`` command
~~~~~~~~~~~~~~~~~~~~

If you want to change your workflow but *keep* the existing processing, you can "fork"
the current working directory and start the new workflow from wherever it diverges from
the original. To do this, use::

    $ edges-analysis fork new-workflow.yaml ./my_workflow --output ./new_workflow

Then, run the ``process`` command as normal with ``--output ./new_workflow``.

Commands
~~~~~~~~

Here, we give a basic overview of the commands available, and their respective options.
Note that ``--help`` can be used at any time on the command line for any command.

.. click:: edges_analysis.cli:process
   :prog: process
   :nested: full

.. click:: edges_analysis.cli:avail
   :prog: avail
   :nested: full

Example Workflow File
~~~~~~~~~~~~~~~~~~~~~
Here is a sample "real-world" workflow file:

.. literalinclude:: workflow.yaml
   :language: yaml
   :caption: An example workflow YAML.
