"""Tests for configuration components."""

import subprocess
import sys
from pathlib import Path


def test_config_module_syntax():
    """Test that config module has valid syntax."""
    project_root = Path(__file__).parent.parent.parent
    config_file = project_root / "ml" / "config.py"

    with open(config_file, "r") as f:
        code = f.read()

    # This will raise SyntaxError if there are syntax issues
    compile(code, str(config_file), "exec")


def test_config_components_available():
    """Test that config components can be imported."""
    project_root = Path(__file__).parent.parent.parent

    test_code = """
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './src')

# Test import of config classes
from ml.config import GBMConfig, HedgingConfig, StressTestConfig

# Test basic instantiation
gbm_cfg = GBMConfig()
hedge_cfg = HedgingConfig()
stress_cfg = StressTestConfig()

print('Config components imported and instantiated successfully')
"""

    result = subprocess.run(
        [sys.executable, "-c", test_code],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Config test failed: {result.stderr}"
