"""Test the actual CLI commands."""


def test_extract_command_does_not_crash(runner, cli_app):
    """Test that the 'extract' command can be called without crashing."""
    result = runner.invoke(cli_app, ["extract", "--help"])
    assert result.exit_code == 0, f"Command crashed with error: {result.exception}"
    assert "Usage" in result.stdout, "Help message not displayed correctly"
