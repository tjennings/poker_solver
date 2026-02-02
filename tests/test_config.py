"""Tests for config loader module."""

import pytest
import tempfile
import os
from pathlib import Path

from config.loader import Config, load_config, get_preset_path


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_creation(self):
        """Test basic Config creation with stack_depths list."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[2.5, 3, 8])
        assert config.name == "Test"
        assert config.stack_depths == [100]
        assert config.stack_depth == 100  # Backward-compatible property
        assert config.raise_sizes == [2.5, 3, 8]

    def test_config_creation_multi_stack(self):
        """Test Config creation with multiple stack depths."""
        config = Config(name="Test", stack_depths=[25, 50, 100], raise_sizes=[2.5, 3, 8])
        assert config.stack_depths == [25, 50, 100]
        assert config.stack_depth == 25  # Returns first stack

    def test_legal_raises_at_open(self):
        """Test get_legal_raise_sizes when opening (current_bet=0)."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[2.5, 3, 8, 20, 50, 100, 200])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=100)
        assert legal == [2.5, 3, 8, 20, 50, 100]  # 200 exceeds stack

    def test_legal_raises_facing_bet(self):
        """Test get_legal_raise_sizes when facing a bet."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[2.5, 3, 8, 20])
        legal = config.get_legal_raise_sizes(current_bet=8, stack=100)
        assert legal == [20]  # Only 20 > 8

    def test_legal_raises_all_below_current_bet(self):
        """Test when all raise sizes are below current bet."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[2.5, 3, 8])
        legal = config.get_legal_raise_sizes(current_bet=10, stack=100)
        assert legal == []

    def test_legal_raises_all_above_stack(self):
        """Test when all raise sizes exceed remaining stack."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[50, 100, 200])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=40)
        assert legal == []

    def test_legal_raises_exact_stack(self):
        """Test that raise size equal to stack is included."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[50, 100])
        legal = config.get_legal_raise_sizes(current_bet=0, stack=100)
        assert legal == [50, 100]

    def test_legal_raises_equal_to_current_bet_excluded(self):
        """Test that raise size equal to current bet is excluded."""
        config = Config(name="Test", stack_depths=[100], raise_sizes=[8, 20])
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
        """Test loading config with all fields specified (legacy format)."""
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
                assert config.stack_depths == [50]
                assert config.raise_sizes == [2, 3, 4]
            finally:
                os.unlink(f.name)

    def test_load_config_with_stack_depths(self):
        """Test loading config with new stack_depths format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "Multi Stack"
stack_depths: [25, 50, 100]
raise_sizes: [2, 3, 8]
""")
            f.flush()
            try:
                config = load_config(f.name)
                assert config.name == "Multi Stack"
                assert config.stack_depths == [25, 50, 100]
                assert config.stack_depth == 25  # First stack
                assert config.raise_sizes == [2, 3, 8]
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
        """Test that missing stack_depth/stack_depths raises ValueError."""
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


class TestConfigValidationErrors:
    """Tests for config validation error cases."""

    def test_malformed_yaml_syntax(self):
        """Test that malformed YAML raises ValueError with clear message."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
name: "Test"
stack_depth: 100
raise_sizes: [2, 3
  - invalid: indentation
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="Invalid YAML syntax"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_non_dict_yaml_number(self):
        """Test that YAML containing just a number raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("42")
            f.flush()
            try:
                with pytest.raises(ValueError, match="must contain a YAML mapping"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_non_dict_yaml_list(self):
        """Test that YAML containing a list raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
- item1
- item2
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="must contain a YAML mapping"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_non_numeric_stack_depth(self):
        """Test that non-numeric stack_depth raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: "not a number"
raise_sizes: [2, 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="stack_depth must be a number"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_non_list_raise_sizes(self):
        """Test that non-list raise_sizes raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 100
raise_sizes: "not a list"
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="raise_sizes must be a list"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_negative_stack_depth(self):
        """Test that negative stack_depth raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: -50
raise_sizes: [2, 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="stack_depth must be positive"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_zero_stack_depth(self):
        """Test that zero stack_depth raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 0
raise_sizes: [2, 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match="stack_depth must be positive"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_negative_raise_size(self):
        """Test that negative raise size raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 100
raise_sizes: [2, -3, 5]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match=r"raise_sizes\[1\] must be positive"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_zero_raise_size(self):
        """Test that zero raise size raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 100
raise_sizes: [0, 2, 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match=r"raise_sizes\[0\] must be positive"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)

    def test_non_numeric_raise_size(self):
        """Test that non-numeric raise size raises ValueError."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
stack_depth: 100
raise_sizes: [2, "bad", 3]
""")
            f.flush()
            try:
                with pytest.raises(ValueError, match=r"raise_sizes\[1\] must be a number"):
                    load_config(f.name)
            finally:
                os.unlink(f.name)
