"""Tests for config loader module."""

import pytest
import tempfile
import os
from pathlib import Path

from config.loader import Config, load_config, get_preset_path


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_creation(self):
        """Test basic Config creation."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8])
        assert config.name == "Test"
        assert config.stack_depth == 100
        assert config.raise_sizes == [2.5, 3, 8]

    def test_legal_raises_at_open(self):
        """Test get_legal_raise_sizes when opening (current_bet=0)."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8, 20, 50, 100, 200])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=100)
        assert legal == [2.5, 3, 8, 20, 50, 100]  # 200 exceeds stack

    def test_legal_raises_facing_bet(self):
        """Test get_legal_raise_sizes when facing a bet."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8, 20])
        legal = config.get_legal_raise_sizes(current_bet=8, stack=100)
        assert legal == [20]  # Only 20 > 8

    def test_legal_raises_all_below_current_bet(self):
        """Test when all raise sizes are below current bet."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[2.5, 3, 8])
        legal = config.get_legal_raise_sizes(current_bet=10, stack=100)
        assert legal == []

    def test_legal_raises_all_above_stack(self):
        """Test when all raise sizes exceed remaining stack."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[50, 100, 200])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=40)
        assert legal == []

    def test_legal_raises_exact_stack(self):
        """Test that raise size equal to stack is included."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[50, 100])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=100)
        assert legal == [50, 100]

    def test_legal_raises_equal_to_current_bet_excluded(self):
        """Test that raise size equal to current bet is excluded."""
        config = Config(name="Test", stack_depth=100, raise_sizes=[8, 20])
        legal = config.get_legal_raise_sizes(current_bet=8, stack=100)
        assert legal == [20]  # 8 is not > 8, so excluded


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_preset_standard(self):
        """Test loading the standard preset."""
        config = load_config(get_preset_path("standard"))
        assert config.name == "Standard 100BB"
        assert config.stack_depth == 100
        assert 2.5 in config.raise_sizes

    def test_load_config_with_all_fields(self):
        """Test loading config with all fields specified."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "Test Config"
stack_depth: 50
raise_sizes: [2, 3, 4]
""")
            f.flush()
            try:
                config = load_config(f.name)
                assert config.name == "Test Config"
                assert config.stack_depth == 50
                assert config.raise_sizes == [2, 3, 4]
            finally:
                os.unlink(f.name)

    def test_load_config_default_name(self):
        """Test that name defaults to 'Custom' if missing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 75
raise_sizes: [2.5, 5, 10]
""")
            f.flush()
            try:
                config = load_config(f.name)
                assert config.name == "Custom"
                assert config.stack_depth == 75
            finally:
                os.unlink(f.name)

    def test_validation_missing_stack(self):
        """Test that missing stack_depth raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "Invalid"
raise_sizes: [2, 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="stack_depth"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_validation_missing_raise_sizes(self):
        """Test that missing raise_sizes raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "Invalid"
stack_depth: 100
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="raise_sizes"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_load_nonexistent_file(self):
        """Test that loading nonexistent file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")


class TestGetPresetPath:
    """Tests for get_preset_path function."""

    def test_get_preset_path_standard(self):
        """Test getting path for standard preset."""
        path = get_preset_path("standard")
        assert path.endswith("config/presets/standard.yaml")
        assert Path(path).exists()

    def test_get_preset_path_returns_absolute(self):
        """Test that get_preset_path returns an absolute path."""
        path = get_preset_path("standard")
        assert Path(path).is_absolute()
