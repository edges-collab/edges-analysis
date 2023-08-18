"""Functions for dealing with CLI workflow and progressfiles.

The progress file is simply a YAML file consisting purely of a list of steps in the
analysis, as defined by a ``workflow.yaml``. Each `step` in the progress file has the
format::

    name: <name of step>
    function: <function that was run at this step>
    params:
        param1: value
        param2: value
        ...
    inout:
    - - [all input files for object1]
        - [all output files from the input files for object1]
    - - [all input files for object2]
        - [all output files for object2]

The ``inout`` is thus a list of 2-tuples, where each tuple specifies an "object" that
goes into an analysis function (some functions take just one input and yield one output,
while others take in multiple inputs and yield one output, or vice versa).
Thus, the first element of each tuple is a list
of input files to the analysis function, and the second element is a list of output
files generated from those input files.

The first time you run a workflow, you must specify with ``-i`` the input files for the
run. This uses the :class:`ProgressFile.create()` method to create a new progress file
from the workflow, and adds the input files to the `convert` step as *outputs*.

"""
from __future__ import annotations

import attrs
import yaml
from jinja2 import Template
from pathlib import Path
from typing import Any, Union

from .gsdata import GSDATA_PROCESSORS

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

Pathy = Union[str, Path]
FileMap = tuple[list[Pathy], list[Pathy]]


class WorkflowProgressError(RuntimeError):
    """Exception raised when the workflow and progress files are discrepant."""

    pass


@attrs.define()
class WorkflowStep:
    function: str = attrs.field(converter=str.lower)
    name: str = attrs.field(converter=str.lower)
    params: dict = attrs.field(factory=dict)
    filemap: list[FileMap] = attrs.field(factory=list)

    @name.default
    def _default_name(self):
        return self.function

    @function.validator
    def _validate_function(self, attribute, value):
        if value != "convert" and value not in GSDATA_PROCESSORS:
            raise ValueError(
                f"Unknown function {value}. Available: {GSDATA_PROCESSORS.keys()}"
            )

    def get_all_outputs(self) -> list[Pathy]:
        """Get a list of output files from a step."""
        return sum((p[1] for p in self.filemap), []) if self.filemap else []

    def get_all_inputs(self) -> list[Pathy]:
        """Get a list of input files from a step."""
        return sum((p[0] for p in self.filemap), []) if self.filemap else []

    def asdict(self, files: bool = True) -> dict[str, Any]:
        """Get the step as a dictionary."""
        d = attrs.asdict(self)
        if not files:
            d.pop("filemap")
        return d

    def compat(self, other: WorkflowStep) -> bool:
        """Check if two steps are compatible."""
        return self.asdict(files=False) == other.asdict(files=False)

    def __call__(self):
        """Call the step function."""
        return GSDATA_PROCESSORS[self.function](**self.params)


@attrs.define()
class Workflow:
    steps: list[WorkflowStep] = attrs.field(factory=list)

    @classmethod
    def read(cls, workflow: Pathy) -> Self:
        """Read a workflow from a file."""
        with open(workflow) as fl:
            workflowd = yaml.load(fl, Loader=yaml.FullLoader)

        global_params = workflowd.pop("globals", {})

        with open(workflow) as fl:
            txt = Template(fl.read())
            txt = txt.render(globals=global_params)
            workflow = yaml.load(txt, Loader=yaml.FullLoader)

        steps = workflow.pop("steps")
        steps = [WorkflowStep(**step) for step in steps]

        all_names = [step.name for step in steps]
        for name in all_names:
            if all_names.count(name) > 1:
                raise ValueError(
                    f"Duplicate step name {name}. "
                    "Please give one of the steps an explicit 'name'."
                )

        return cls(steps=steps)

    def write_as_progressfile(self, progressfile: Pathy):
        """Write the workflow as a progressfile."""
        progress = [attrs.asdict(step) for step in self.steps]

        with open(progressfile, "w") as fl:
            yaml.dump(progress, fl)

    def __getitem__(self, key: int | str):
        """Get a step from the workflow."""
        if isinstance(key, int):
            return self.steps[key]
        elif isinstance(key, str):
            return self.steps[[s.name for s in self.steps].index(key)]

    def __contains__(self, key: str) -> bool:
        """Check if a step is in the workflow."""
        return key in [s.name for s in self.steps]

    def __iter__(self):
        """Iterate over the steps in the workflow."""
        return iter(self.steps)

    def append(self, step: WorkflowStep):
        """Append a step to the workflow."""
        self.steps.append(step)

    def index(self, key: str) -> int:
        """Get the index of a step."""
        return [s.name for s in self.steps].index(key)


@attrs.define()
class ProgressFile:
    path: Path = attrs.field(
        converter=Path,
        validator=attrs.validators.and_(
            attrs.validators.instance_of(Path),
            attrs.validators.file_exists(),
        ),
    )
    workflow: Workflow = attrs.field(factory=list)

    @classmethod
    def create(cls, progressfile: Pathy, workflow: Workflow, inputs: list[Path] = None):
        """Create a new progressfile."""
        if any(bool(s.filemap) for s in workflow.steps):
            raise ValueError(
                "Cannot create a new progressfile for a workflow with filemaps "
                "already set."
            )

        # Now, add the inputs to the convert step.
        if "convert" in workflow:
            # TODO: is this really right?
            workflow["convert"].filemap.append(([], [str(p) for p in inputs]))

        workflow.write_as_progressfile(progressfile)

        return cls(path=progressfile, workflow=workflow)

    @classmethod
    def read(cls, progressfile: Pathy) -> Self:
        """Read the progressfile."""
        with open(progressfile) as openfile:
            progress = yaml.load(openfile, Loader=yaml.FullLoader)

        progress = Workflow([WorkflowStep(**p) for p in progress])
        return cls(path=progressfile, workflow=progress)

    def __getitem__(self, key: int | str):
        """Get a step from the workflow."""
        return self.workflow[key]

    def __contains__(self, key: str) -> bool:
        """Check if a step is in the workflow."""
        return key in self.workflow

    def __iter__(self):
        """Iterate over the steps in the workflow."""
        return iter(self.workflow)

    def update_step(self, key: str, filemap: FileMap):
        """Update the progress file with a new filemap for a step."""
        if key not in self:
            raise ValueError(f"Progress file has no step called '{key}'")

        self[key].filemap.extend(filemap)

        if key == "convert":
            # We also need to remove files that are gotten by combining datafiles,
            # because if we're adding new inputs, these files will end up changing.
            blastoff = False
            for step in self:
                if step.name == "convert":
                    continue

                if GSDATA_PROCESSORS[step.function].kind == "gather":
                    blastoff = True

                if blastoff:
                    for fl in step.get_all_outputs():
                        if Path(fl).exists():
                            Path(fl).unlink()
                    step.filemap.clear()

        self.workflow.write_as_progressfile(self.path)

    def harmonize_with_workflow(
        self,
        workflow: Workflow,
        error: bool = True,
        start: str = None,
    ) -> Self:
        """Check the compatibility of the current steps with the progressfile."""
        start_changing = False
        for i, step in enumerate(workflow):
            if i >= len(self):
                # This is the case when new steps are added to the workflow since
                # last run.
                self.workflow.append(step)
            else:
                ps = self[i]
                if not ps.compat(step):
                    start_changing = True
                    if error:
                        raise WorkflowProgressError(
                            "The workflow is in conflict with the progressfile at step "
                            f"'{step.name}'. To remove conflicting outputs and adopt "
                            "new workflow, run with --ignore-conflicting. To keep the "
                            "existing outputs and branch off with the new workflow, run"
                            " the 'fork' command."
                        )
                if step.name == start:
                    start_changing = True

                if start_changing:
                    outputs = ps.get_all_outputs()
                    for fl in outputs:
                        # ensure file is in outdir. Raw input files (eg. ACQ files)
                        # should not be deleted.
                        if str(self.path.parent) in str(fl):
                            Path(fl).unlink(missing_ok=True)
                    ps.filemap.clear()

        self.workflow.write_as_progressfile(self.path)

    def get_files_to_read_for_step(self, stepname: str) -> list[str]:
        """Get all the files we need to read for a given step."""
        # First, get most recent outputs.
        current_index = self.workflow.index(stepname)

        potential_files = self.workflow[current_index].get_all_outputs()
        final_files = []

        def _check_fl(fl):
            # Check if an output file (fl) for the current step appears as an input file
            # for a later step, and whether all the output files from that step exist.
            for i, p in enumerate(self):
                if i <= current_index:
                    continue

                inp = p.get_all_inputs()
                out = p.get_all_outputs()

                if fl in inp and all(Path(x).exists() for x in out):
                    return False
            return True

        for fl in potential_files:
            if _check_fl(fl):
                final_files.append(fl)

        return [fl for fl in potential_files if _check_fl(fl)]
