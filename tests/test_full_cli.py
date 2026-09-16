"""T4: `full` 命令注册且签名可用（不跑真数据，e2e 见 /tmp/full_run 手动验证）。"""
from typer.testing import CliRunner

from cercus.cli.app import app


def test_full_command_registered():
    names = {c.callback.__name__ for c in app.registered_commands}
    assert "full" in names
    res = CliRunner().invoke(app, ["full", "--help"])
    assert res.exit_code == 0
    assert "--workers" in res.output
