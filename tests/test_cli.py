import pytest
from typer.testing import CliRunner

from judgecheck.cli import app

runner = CliRunner()


def test_help() -> None:
    assert runner.invoke(app, ["--help"]).exit_code == 0


@pytest.mark.parametrize("command", ["audit", "probe", "report"])
def test_commands_not_implemented(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == 0
    assert "not implemented yet" in result.output
