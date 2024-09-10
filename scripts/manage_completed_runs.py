#!/usr/bin/env python3
"""
Script to manage completed training runs.
This script provides utilities to view and manage the completion status of training runs.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import load_config
from src.training.manager import (
    list_completed_runs, 
    print_completed_runs_summary, 
    clear_completion_status,
    load_completion_status
)


def main():
    parser = argparse.ArgumentParser(description="Manage completed training runs")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yml",
        help="Path to the configuration file"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all completed runs')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear completion status for a run')
    clear_parser.add_argument('run_id', help='Run ID to clear completion status for')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show status of a specific run')
    status_parser.add_argument('run_id', help='Run ID to check status for')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Load configuration
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    if args.command == 'list':
        print_completed_runs_summary(cfg)
        
    elif args.command == 'clear':
        success = clear_completion_status(args.run_id, cfg)
        if success:
            print(f"Successfully cleared completion status for run: {args.run_id}")
        else:
            print(f"Failed to clear completion status for run: {args.run_id}")
            
    elif args.command == 'status':
        status = load_completion_status(args.run_id, cfg)
        if status:
            print(f"\nCompletion status for run {args.run_id}:")
            print(f"  Completed: {status['completed']}")
            print(f"  Reason: {status['completion_reason']}")
            print(f"  Final Step: {status['final_step']}")
            print(f"  Final Best Loss: {status['final_best_loss']}")
            print(f"  Completion Time: {status['completion_timestamp']}")
            if 'post_training_eval_done' in status:
                print(f"  Post-training Eval Done: {status['post_training_eval_done']}")
            if 'best_checkpoint_logged' in status:
                print(f"  Best Checkpoint Logged: {status['best_checkpoint_logged']}")
        else:
            print(f"No completion status found for run: {args.run_id}")


if __name__ == "__main__":
    main() 