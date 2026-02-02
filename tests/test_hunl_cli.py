"""Tests for HUNL CLI subcommand."""

import subprocess
import sys
import pytest


class TestHUNLCLIHelp:
    """Tests for HUNL CLI help and usage."""

    def test_hunl_help(self):
        """The hunl subcommand should show help."""
        result = subprocess.run(
            [sys.executable, "main.py", "hunl", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "hunl" in result.stdout.lower()
        assert "--config" in result.stdout
        assert "--iterations" in result.stdout

    def test_hunl_requires_config(self):
        """The hunl subcommand should require --config."""
        result = subprocess.run(
            [sys.executable, "main.py", "hunl"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        # Should mention config is required
        assert "config" in result.stderr.lower() or "required" in result.stderr.lower()


class TestHUNLCLIPresets:
    """Tests for HUNL CLI with preset configs."""

    def test_hunl_with_tiny_preset(self):
        """HUNL should run with tiny preset (fast test)."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "tiny",
                "--iterations",
                "10",
                "--batch-size",
                "1",
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_hunl_with_invalid_preset(self):
        """HUNL should fail with invalid preset name."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "nonexistent_preset",
                "--iterations",
                "10",
                "--no-interactive",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestHUNLCLIOptions:
    """Tests for HUNL CLI options."""

    def test_hunl_device_option(self):
        """HUNL should accept device option."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "tiny",
                "--iterations",
                "10",
                "--device",
                "cpu",
                "--batch-size",
                "1",
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    @pytest.mark.slow
    def test_hunl_batch_size_option(self):
        """HUNL should accept batch-size option.

        This test is slow because batched mode requires compiling the
        entire game tree to tensors.
        """
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "tiny",
                "--iterations",
                "100",
                "--batch-size",
                "512",
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes for batched mode compilation
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_hunl_with_action_sequence(self):
        """HUNL should accept action sequence."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "tiny",
                "--iterations",
                "10",
                "--batch-size",
                "1",
                "--action",
                "SBr2.5",
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_hunl_with_stack_override_in_action(self):
        """HUNL should accept stack override in action sequence."""
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "hunl",
                "--config",
                "tiny",
                "--iterations",
                "10",
                "--batch-size",
                "1",
                "--action",
                "5bb SBr2.5",
                "--no-interactive",
                "--quiet",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"


class TestKuhnCommandBackwardCompatibility:
    """Tests to ensure kuhn command still works."""

    def test_kuhn_command_exists(self):
        """Kuhn command should still work."""
        result = subprocess.run(
            [sys.executable, "main.py", "kuhn", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--iterations" in result.stdout

    def test_kuhn_as_default(self):
        """Default command (no subcommand) should run kuhn."""
        result = subprocess.run(
            [sys.executable, "main.py", "--iterations", "100", "--quiet"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Kuhn" in result.stdout
