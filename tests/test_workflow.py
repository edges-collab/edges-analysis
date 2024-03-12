"""Test of the _workflow module."""

from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import pytest
import yaml
from edges_analysis import _workflow as wf
from edges_analysis.gsdata.select import select_freqs


class TestFileMapEntry:
    # Create a FileMapEntry object with valid inputs and outputs.
    def test_create_valid_inputs_outputs(self):
        inputs = frozenset([Path("input1.txt"), Path("input2.txt")])
        outputs = frozenset([Path("output1.txt"), Path("output2.txt")])
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert entry.inputs == inputs
        assert entry.outputs == outputs

    # Two FileMapEntry objects with the same inputs and outputs should be equal.
    def test_equality(self):
        inputs = frozenset([Path("input1.txt"), Path("input2.txt")])
        outputs = frozenset([Path("output1.txt"), Path("output2.txt")])
        entry1 = wf.FileMapEntry(inputs=inputs, outputs=outputs)
        entry2 = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert entry1 == entry2

    # Hash of two FileMapEntry objects with the same inputs and outputs should be equal.
    def test_hash_equality(self):
        inputs = frozenset([Path("input1.txt"), Path("input2.txt")])
        outputs = frozenset([Path("output1.txt"), Path("output2.txt")])
        entry1 = wf.FileMapEntry(inputs=inputs, outputs=outputs)
        entry2 = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert hash(entry1) == hash(entry2)

    # Create a FileMapEntry object with empty inputs and outputs.
    def test_create_empty_inputs_outputs(self):
        inputs = frozenset()
        outputs = frozenset()
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert entry.inputs == inputs
        assert entry.outputs == outputs

    # Create a FileMapEntry object with a single input and output.
    def test_create_single_input_output(self):
        inputs = frozenset([Path("input.txt")])
        outputs = frozenset([Path("output.txt")])
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert entry.inputs == inputs
        assert entry.outputs == outputs

    # Create a FileMapEntry object with non-unique inputs or outputs.
    def test_create_non_unique_inputs_outputs(self):
        inputs = frozenset([Path("input.txt"), Path("input.txt")])
        outputs = frozenset([Path("output.txt"), Path("output.txt")])
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert entry.inputs == inputs
        assert entry.outputs == outputs

    # Create a FileMapEntry object with non-frozen inputs.
    def test_create_non_frozen_inputs(self):
        inputs = {"input1.txt", "input2.txt"}
        outputs = {"output1.txt", "output2.txt"}
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert isinstance(entry.inputs, frozenset)
        assert isinstance(entry.outputs, frozenset)
        assert all(isinstance(x, Path) for x in entry.inputs)
        assert all(isinstance(x, Path) for x in entry.outputs)

    # Test the asdict method
    def test_asdict(self):
        inputs = frozenset([Path("input1.txt"), Path("input2.txt")])
        outputs = frozenset([Path("output1.txt"), Path("output2.txt")])
        entry = wf.FileMapEntry(inputs=inputs, outputs=outputs)

        assert isinstance(entry.asdict()["inputs"][0], str)
        assert isinstance(entry.asdict()["outputs"][0], str)


@pytest.fixture()
def file_map() -> wf.FileMap:
    return wf.FileMap(
        (
            (["input1.txt", "input2.txt"], ["output1.txt", "output2.txt"]),
            (["input3.txt", "input4.txt"], ["output3.txt", "output4.txt"]),
        )
    )


class TestFileMap:
    def test_create_empty(self):
        file_map = wf.FileMap()
        assert file_map.maps == set()

    def test_create_from_dicts(self):
        file_map = wf.FileMap(
            (
                {
                    "inputs": ["input1.txt", "input2.txt"],
                    "outputs": ["output1.txt", "output2.txt"],
                },
                {
                    "inputs": ["input3.txt", "input4.txt"],
                    "outputs": ["output3.txt", "output4.txt"],
                },
            )
        )

        assert len(file_map) == 2
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map)

    def test_create_from_tuples(self, file_map: wf.FileMap):
        assert len(file_map) == 2
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map)

    def test_create_from_entries(self):
        file_map = wf.FileMap(
            (
                wf.FileMapEntry(
                    inputs=frozenset(["input1.txt", "input2.txt"]),
                    outputs=frozenset(["output1.txt", "output2.txt"]),
                ),
                wf.FileMapEntry(
                    inputs=frozenset(["input3.txt", "input4.txt"]),
                    outputs=frozenset(["output3.txt", "output4.txt"]),
                ),
            )
        )

        assert len(file_map.maps) == 2
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map.maps)

    def test_iterable_properties(self, file_map: wf.FileMap):
        assert len(file_map) == 2
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map)
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map.maps)
        assert bool(file_map)

        file_map.clear()
        assert len(file_map) == 0

    def test_remove_file(self, file_map: wf.FileMap):
        file_map.remove("input1.txt")
        assert len(file_map) == 1
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map)
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map.maps)
        assert bool(file_map)

    def test_add_maps(self, file_map: wf.FileMap):
        file_map.add(
            (
                (["input5.txt", "input6.txt"], ["output5.txt", "output6.txt"]),
                (["input7.txt", "input8.txt"], ["output7.txt", "output8.txt"]),
            )
        )

        assert len(file_map) == 4
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map)
        assert all(isinstance(x, wf.FileMapEntry) for x in file_map.maps)
        assert bool(file_map)

    def test_as_yamlable(self, file_map: wf.FileMap):
        yamlable = file_map.as_yamlable()
        assert isinstance(yamlable, list)
        assert all(isinstance(x, dict) for x in yamlable)

        # Simply test that it actually dumps.
        asstring = yaml.dump(yamlable)

        # Test that it can be loaded back in.
        loaded = yaml.load(asstring, Loader=yaml.SafeLoader)

        # Test that the loaded object is the same as the original.
        assert loaded == yamlable


@pytest.fixture()
def empty_step() -> wf.WorkflowStep:
    return wf.WorkflowStep("convert")


@pytest.fixture()
def map_step(file_map: wf.FileMap) -> wf.WorkflowStep:
    return wf.WorkflowStep("convert", filemap=file_map)


class TestWorkflowStep:
    def test_create_badly(self):
        with pytest.raises(ValueError, match="Unknown function non-existent"):
            wf.WorkflowStep(function="non-existent")

    def test_default_name(self, empty_step: wf.WorkflowStep):
        assert empty_step.name == "convert"

    def test_get_all_outputs_empty(self, empty_step: wf.WorkflowStep):
        assert empty_step.get_all_outputs() == frozenset()

    def test_get_all_outputs_full(self, map_step: wf.WorkflowStep):
        assert map_step.get_all_outputs() == frozenset(
            [
                Path("output1.txt"),
                Path("output2.txt"),
                Path("output3.txt"),
                Path("output4.txt"),
            ]
        )

    def test_get_all_inputs_empty(self, empty_step: wf.WorkflowStep):
        assert empty_step.get_all_inputs() == frozenset()

    def test_get_all_inputs_full(self, map_step: wf.WorkflowStep):
        assert map_step.get_all_inputs() == frozenset(
            [
                Path("input1.txt"),
                Path("input2.txt"),
                Path("input3.txt"),
                Path("input4.txt"),
            ]
        )

    def test_asdict(self, empty_step: wf.WorkflowStep):
        asdict = empty_step.asdict()
        assert isinstance(asdict, dict)
        assert asdict["name"] == "convert"
        assert asdict["function"] == "convert"
        assert not asdict["filemap"]

    def test_asdict_nofiles(self, map_step: wf.WorkflowStep):
        asdict = map_step.asdict(files=False)
        assert isinstance(asdict, dict)
        assert asdict["name"] == "convert"
        assert asdict["function"] == "convert"
        assert "filemap" not in asdict

    def test_compat(self, empty_step, map_step):
        assert empty_step.compat(empty_step)
        assert empty_step.compat(map_step)
        assert map_step.compat(empty_step)
        assert map_step.compat(map_step)

    def test_auto_function(self):
        step = wf.WorkflowStep(function="select_freqs")
        assert step._function == select_freqs
        assert step.kind == "reduce"

    def test_get_output_path(self):
        step = wf.WorkflowStep(function="select_freqs")

        outfile = step.get_output_path(
            outdir=".",
            infile=Path("input.gsh5"),
        )
        assert outfile is None

        step = wf.WorkflowStep(function="select_freqs", write="output.gsh5")
        outfile = step.get_output_path(
            outdir=".",
            infile=Path("input.gsh5"),
        )
        assert outfile == Path("output.gsh5")

        step = wf.WorkflowStep(function="select_freqs", write="/output.gsh5")
        outfile = step.get_output_path(
            outdir="output",
            infile=Path("input.gsh5"),
        )

        assert outfile == Path("/output.gsh5")

        step = wf.WorkflowStep(function="select_freqs", write="output.{fncname}.gsh5")
        outfile = step.get_output_path(
            outdir=".",
            infile=Path("input.gsh5"),
        )
        assert outfile == Path("output.select_freqs.gsh5")

    def test_has_input(self, map_step: wf.WorkflowStep):
        assert map_step.has_input("input1.txt")
        assert map_step.has_input("input2.txt")
        assert map_step.has_input("input3.txt")
        assert map_step.has_input("input4.txt")
        assert not map_step.has_input("input5.txt")

    def test_has_output(self, map_step: wf.WorkflowStep):
        assert map_step.has_output("output1.txt")
        assert map_step.has_output("output2.txt")
        assert map_step.has_output("output3.txt")
        assert map_step.has_output("output4.txt")
        assert not map_step.has_output("output5.txt")

    def test_get_outputs_for_input(self, map_step: wf.WorkflowStep):
        assert map_step.get_outputs_for_input("input1.txt") == frozenset(
            [Path("output1.txt"), Path("output2.txt")]
        )
        assert map_step.get_outputs_for_input("input2.txt") == frozenset(
            [Path("output1.txt"), Path("output2.txt")]
        )
        assert map_step.get_outputs_for_input("input3.txt") == frozenset(
            [Path("output3.txt"), Path("output4.txt")]
        )
        assert map_step.get_outputs_for_input("input4.txt") == frozenset(
            [Path("output3.txt"), Path("output4.txt")]
        )
        assert map_step.get_outputs_for_input("input5.txt") == frozenset()

    def test_add_to_filemap(self, empty_step, file_map):
        empty_step.add_to_filemap(file_map)
        assert len(empty_step.filemap) == 2

    def test_remove_from_filemap(self, map_step):
        map_step.remove_from_filemap("input1.txt")
        assert len(map_step.filemap) == 1


@pytest.fixture()
def simple_workflow_yaml(tmp_path) -> Path:
    out = tmp_path / "workflow.yaml"

    with out.open("w") as fl:
        fl.write(
            """
        steps:
          - function: convert
          - name: select
            function: select_freqs
            write: output.gsh5
        """
        )
    return out


@pytest.fixture()
def simple_workflow(simple_workflow_yaml) -> wf.Workflow:
    return wf.Workflow.read(simple_workflow_yaml)


class TestWorkflow:
    def test_read(self, simple_workflow):
        new = wf.Workflow(
            steps=[
                wf.WorkflowStep("convert"),
                wf.WorkflowStep("select_freqs", name="select", write="output.gsh5"),
            ]
        )
        assert new == simple_workflow

    def test_getitem(self, simple_workflow):
        assert simple_workflow["convert"].name == "convert"
        assert simple_workflow["select"].name == "select"
        assert simple_workflow[0] == simple_workflow["convert"]
        assert simple_workflow[1] == simple_workflow["select"]

    def test_setitem_badtype(self, simple_workflow):
        with pytest.raises(TypeError):
            simple_workflow["convert"] = 1

        with pytest.raises(TypeError):
            simple_workflow[27.8] = wf.WorkflowStep("select_lsts")

    def test_setitem_badname(self, simple_workflow):
        with pytest.raises(ValueError, match="Duplicate step name select"):
            simple_workflow["convert"] = simple_workflow["select"]

    def test_setitem(self, simple_workflow):
        simple_workflow["select"] = wf.WorkflowStep("select_lsts")
        assert len(simple_workflow) == 2

        simple_workflow[1] = wf.WorkflowStep("select_freqs")
        assert simple_workflow[1].name == "select_freqs"

    def test_contains(self, simple_workflow):
        assert "convert" in simple_workflow

    def test_append(self, simple_workflow):
        simple_workflow.append(wf.WorkflowStep("select_lsts"))
        assert len(simple_workflow) == 3

    def test_index(self, simple_workflow):
        assert simple_workflow.index("convert") == 0
        assert simple_workflow.index("select") == 1

    def test_insert(self, simple_workflow):
        simple_workflow.insert(1, wf.WorkflowStep("select_lsts"))
        assert len(simple_workflow) == 3
        assert simple_workflow.index("select") == 2


@pytest.fixture()
def simple_progressfile_yaml(simple_workflow, tmp_path):
    out = tmp_path / "progressfile.yaml"

    simple_workflow.write_as_progressfile(out)
    return out


@pytest.fixture()
def empty_progressfile(simple_progressfile_yaml) -> wf.ProgressFile:
    return wf.ProgressFile.read(simple_progressfile_yaml)


@pytest.fixture()
def progressfile(empty_progressfile) -> wf.ProgressFile:
    empty_progressfile.add_inputs(["input1.txt", "input2.txt"])
    return empty_progressfile


class TestProgressFile:
    def test_create_badly(self):
        with pytest.raises(ValueError, match="does not exist"):
            wf.ProgressFile("non-existent.yaml")

    def test_add_inputs_nonunique(self, progressfile):
        progressfile.add_inputs(["input1.txt", "input2.txt"])
        assert len(progressfile["convert"].get_all_inputs()) == 2

    def test_create_with_existing_filemap(self, simple_workflow):
        simple_workflow["convert"].add_to_filemap([[("input1.txt",), ()]])
        with pytest.raises(ValueError, match="Cannot create a new progressfile"):
            wf.ProgressFile.create("progressfile.yaml", workflow=simple_workflow)

    def test_create(self, tmp_path: Path, simple_workflow: wf.Workflow):
        assert not simple_workflow["convert"].has_input("input1.txt")
        prg = wf.ProgressFile.create(
            tmp_path / "_progressfile.yaml",
            workflow=simple_workflow,
            inputs=["input1.txt", "input2.txt"],
        )
        assert prg.has_input("input1.txt")
        assert not simple_workflow["convert"].has_input("input1.txt")

    def test_magics(self, progressfile: wf.ProgressFile):
        assert len(progressfile) == 2
        assert progressfile[0].name == "convert"
        assert progressfile[1].name == "select"
        assert "convert" in progressfile

    def test_remove_inputs(self, progressfile: wf.ProgressFile):
        progressfile.remove_inputs(["input1.txt"])
        assert progressfile.workflow["convert"].has_input("input2.txt")
        assert not progressfile.workflow["convert"].has_input("input1.txt")

        progressfile.remove_inputs(["non-existent.txt"])
        assert progressfile.workflow["convert"].has_input("input2.txt")

    def test_update_step(self, progressfile: wf.ProgressFile):
        copy = progressfile.path.with_suffix(".tmp.yaml")
        copyfile(progressfile.path, copy)
        new = wf.ProgressFile.read(copy)

        new.update_step("select", [[("input1.txt",), ("output1.txt",)]])
        assert new["select"].has_output("output1.txt")

        with pytest.raises(ValueError, match="Progress file has no step"):
            new.update_step("non-existent", ())

    def test_harmonize_no_change(
        self, progressfile: wf.ProgressFile, simple_workflow: wf.Workflow
    ):
        copy = deepcopy(progressfile)

        progressfile.harmonize_with_workflow(simple_workflow)

        assert copy == progressfile

    def test_harmonize_change_step(
        self, progressfile: wf.ProgressFile, simple_workflow: wf.Workflow
    ):
        copy = deepcopy(progressfile)

        progressfile.update_step("select", [[("input1.txt",), ("output1.txt",)]])
        assert progressfile["select"].has_output("output1.txt")

        # Change a parameter, but keep step name the same.
        simple_workflow["select"] = wf.WorkflowStep(
            name="select", function="select_freqs", params={"freq_min": 0.1}
        )
        with pytest.raises(wf.WorkflowProgressError):
            progressfile.harmonize_with_workflow(simple_workflow)

        progressfile.harmonize_with_workflow(simple_workflow, error=False)
        assert copy != progressfile
        assert not progressfile["select"].has_output("output1.txt")

        # Change the step name.
        simple_workflow["select"] = wf.WorkflowStep(
            "select_lsts", params={"lst_min": 0.1}
        )
        progressfile.harmonize_with_workflow(simple_workflow, error=False)
        assert copy != progressfile
        assert "select" not in progressfile

    def test_harmonize_with_start(
        self, progressfile: wf.ProgressFile, simple_workflow: wf.Workflow
    ):
        progressfile.update_step("select", [[("input1.txt",), ("output1.txt")]])
        progressfile.harmonize_with_workflow(simple_workflow, start="convert")
        for step in progressfile:
            assert not step.get_all_outputs()


@pytest.fixture()
def progressfile_extended(tmp_path):
    workflow = tmp_path / "workflow_extended.yaml"

    with workflow.open("w") as fl:
        fl.write(
            """
steps:
    - function: convert
    - function: select_freqs
      write: "{prev_stem}.{fncname}.txt"
    - function: select_lsts
      write: "{prev_stem}.{fncname}.txt"
    - function: lst_average
      write: average.txt
"""
        )
    workflow = wf.Workflow.read(workflow)

    paths = [tmp_path / "input1.txt", tmp_path / "input2.txt"]

    return wf.ProgressFile.create(
        tmp_path / "_progressfile.yaml",
        workflow=workflow,
        inputs=paths,
    )


class TestProgressFileDynamics:
    def mock_run_progressfile(
        self,
        path: Path,  # Progressfile path
        paths: list[Path] = (),  # paths to more inputs, if any
        stop=None,  # step to stop on
    ) -> dict[str, set[Path]]:
        required_files = {}

        prg = wf.ProgressFile.read(path)
        if paths:
            prg.add_inputs(paths)

        # Make the files so they exist
        for p in paths:
            p.touch()

        # Do each step, and write outputs for each
        data = []  # supposed to be the actual in-memory data throughout.
        for step in prg:
            files = prg.get_files_to_read_for_step(step.name)
            files = [f for f in files if f not in data]

            required_files[step.name] = files

            # Mock read-in of the data at this step
            data += list(files)

            if step.write:
                oldfiles = data.copy()
                if step.name == "convert" or step.kind != "combine":
                    data = [step.get_output_path(prg.path.parent, d) for d in data]
                    prg.update_step(
                        step.name, [[(p,), (np,)] for p, np in zip(oldfiles, data)]
                    )

                else:
                    data = [step.get_output_path(prg.path.parent, "")]
                    prg.update_step(step.name, [[oldfiles, data]])

                # Make the file actually exist...
                for p in data:
                    p.touch()

            if step.name == stop:
                break

        return prg, required_files

    def test_add_file_to_complete_run(self, progressfile_extended):
        prg = progressfile_extended

        # Run the whole workflow with the two files
        prg, _ = self.mock_run_progressfile(prg.path)

        # Now, add a new input.
        prg, req = self.mock_run_progressfile(
            prg.path, [prg.path.parent / "input3.txt"], stop="select_lsts"
        )

        # Straight away, we should see that the final combined output no longer exists
        assert not (prg.path.parent / "average.txt").exists()

        # Also, we should have required input3.txt at the convert step.
        # The following steps should NOT require it, because it will already have been
        # read in at that point.
        assert prg.path.parent / "input3.txt" in req["convert"]

        assert not req["select_freqs"]
        assert not req["select_lsts"]

    def test_partial_run(self, progressfile_extended):
        prg = progressfile_extended

        prg, _ = self.mock_run_progressfile(prg.path, stop="select_freqs")

        # Don't add more files, just run to completion.
        prg, req = self.mock_run_progressfile(prg.path)

        # On select_lsts (after select_freqs) we should have had to 'read' the prev
        # files now
        assert prg.path.parent / "input1.select_freqs.txt" in req["select_lsts"]

    def test_add_file_to_partial_run(self, progressfile_extended):
        prg = progressfile_extended

        prg, _ = self.mock_run_progressfile(prg.path, stop="select_freqs")

        # add file and run to completion
        prg, req = self.mock_run_progressfile(
            prg.path, paths=[prg.path.parent / "input3.txt"]
        )

        # now, we should have had to read both of the first two files only at the
        # select_lsts step
        assert prg.path.parent / "input1.select_freqs.txt" in req["select_lsts"]
        assert prg.path.parent / "input2.select_freqs.txt" in req["select_lsts"]
        assert prg.path.parent / "input1.txt" not in req["select_freqs"]
        assert prg.path.parent / "input2.txt" not in req["select_freqs"]

        # However, we should have had to read the third input at the convert step
        assert prg.path.parent / "input3.txt" in req["convert"]
