import pytest
import pkgutil
import sys
import os

# Add src to the path to allow direct import
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

import src


@pytest.mark.smoke
def test_src_imports():
    """
    Walks through all modules in the 'src' package and tries to import them.
    This is a good way to catch basic syntax errors or import issues without
    running the full test suite.
    """
    package = src
    prefix = package.__name__ + "."

    for _, name, ispkg in pkgutil.walk_packages(package.__path__, prefix):
        if not ispkg:
            try:
                __import__(name, fromlist=["__name__"])
            except ImportError as e:
                pytest.fail(f"Failed to import module {name}. Error: {e}")
