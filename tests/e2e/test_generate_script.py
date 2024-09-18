"""
End-to-end tests for generate.py script.

Tests the complete generation script functionality including argument parsing,
config loading, and text generation.
"""

import pytest
import subprocess
import sys
import tempfile
import os


class TestGenerateScript:
    """Test the generate.py script end-to-end."""
    
    @pytest.mark.e2e
    def test_generate_script_help(self):
        """Test that generate.py shows help without errors."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/generate.py", "--help"],
            capture_output=True,
            text=True,
            env=env
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "--config" in result.stdout
    
    @pytest.mark.e2e
    def test_generate_script_with_invalid_config(self):
        """Test generate.py with non-existent config file."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/generate.py", "--config", "nonexistent.yml"],
            capture_output=True,
            text=True,
            env=env
        )
        
        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "not found" in result.stderr.lower()


class TestManageCompletedRunsScript:
    """Test the manage_completed_runs.py script end-to-end."""
    
    @pytest.mark.e2e
    def test_manage_script_help(self):
        """Test that manage_completed_runs.py shows help without errors."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/manage_completed_runs.py", "--help"],
            capture_output=True,
            text=True,
            env=env
        )
        
        assert result.returncode == 0
        assert "usage:" in result.stdout.lower()
        assert "list" in result.stdout
        assert "clear" in result.stdout
        assert "status" in result.stdout
    
    @pytest.mark.e2e
    def test_manage_script_list_command(self):
        """Test the list command with default config."""
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        result = subprocess.run(
            [sys.executable, "scripts/manage_completed_runs.py", "list"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Should complete successfully even if no completed runs exist
        assert result.returncode == 0 