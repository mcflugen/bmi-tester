import contextlib

import pytest

from bmi_tester.main import main


def test_cli_version(capsys):
    with contextlib.suppress(SystemExit):
        assert main(["--version"]) == 0
    output = capsys.readouterr().out

    assert "bmi-test" in output


def test_cli_help(capsys):
    with contextlib.suppress(SystemExit):
        assert main(["--help"]) == 0
    output = capsys.readouterr().out

    assert "usage" in output


@pytest.mark.parametrize(
    "package",
    (
        "testing.points_bmi:PointsBmi",
        "testing.rectilinear_bmi:RectilinearBmi",
        "testing.uniform_rectilinear_bmi:UniformRectilinearBmi",
        "testing.unstructured_bmi:UnstructuredNodesBmi",
        "testing.unstructured_bmi:UnstructuredNodesEdgesBmi",
        "testing.unstructured_bmi:UnstructuredNodesEdgesFacesBmi",
    ),
)
def test_cli_run(capsys, tmpdir, package):
    with tmpdir.as_cwd():
        with open("input.yaml", "w"):
            pass
        with contextlib.suppress(SystemExit):
            assert (
                main([package, f"--root-dir={tmpdir!s}", "--config-file=input.yaml"])
                == 0
            )
