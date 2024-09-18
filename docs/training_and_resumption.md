# Training and Resumption System

This document explains the training lifecycle, smart resumption logic, and completion tracking system in the transformer training pipeline.

## Recent Improvements (Latest Version)

**Fixed Smart Resumption Logic**: The system now properly integrates completion tracking with checkpoint detection:

✅ **Completion Status Priority**: Completed runs are never resumed, even if checkpoints exist  
✅ **Comprehensive Test Coverage**: 14 unit tests cover all resumption scenarios  
✅ **Integration with Existing System**: Works seamlessly with existing completion tracking  
✅ **Intelligent Suffix Handling**: Finds next available run ID or resumes interrupted suffix runs  
✅ **Robust Error Handling**: Handles missing directories, malformed files, and edge cases

**Before**: Smart resumption only checked checkpoint existence, could try to resume completed runs  
**After**: Smart resumption checks completion status first, then checkpoint existence for truly interrupted runs

## Overview

The training system is designed to handle:

- **Automatic resumption** of interrupted training runs
- **Smart conflict avoidance** to prevent overwriting completed runs
- **Persistent completion tracking** that survives checkpoint cleanup
- **Intelligent run ID generation** that balances resumption with new run creation

## Architecture

### Core Components

1. **Smart Run ID Generation** (`generate_smart_run_id()`)
2. **Completion Status Tracking** (`save_completion_status()`, `load_completion_status()`)
3. **Enhanced Metadata Logging** (`add_run_metadata()`)
4. **Checkpoint Management** (automatic cleanup after successful completion)

### Separation of Concerns

- **JSON Metadata Files**: Track completion status (`completion_tracking/<run_id>_completed.json`)
- **Checkpoint File Existence**: Determine resumption vs. new run creation
- **Smart Run ID Generation**: Automatic suffix increment for new runs, same ID for resumption
- **W&B Integration**: Enhanced metadata logging with timestamps and resumption status

## Smart Resumption Logic

### Decision Flow

When starting a training run with base ID `single_bs16_cw128_lr1e-04`:

```
1. Check if base_run_id was completed
   ├─ YES → Create new run with suffix (_2, _3, etc.)
   └─ NO → Check if checkpoints exist
      ├─ YES → Resume with same ID (interrupted run)
      └─ NO → Use base_run_id (fresh start)

2. If creating suffixed run, check each suffix:
   ├─ If suffix run completed → Skip to next suffix
   ├─ If suffix run has checkpoints but not completed → Resume suffix run
   └─ If suffix run has no checkpoints and not completed → Use suffix run
```

### Examples

#### Example 1: Fresh Start

```
State: No checkpoints, no completion status
Input: "single_bs16_cw128_lr1e-04"
Output: ("single_bs16_cw128_lr1e-04", False)
```

#### Example 2: Interrupted Run

```
State: Checkpoints exist, no completion status
Input: "single_bs16_cw128_lr1e-04"
Output: ("single_bs16_cw128_lr1e-04", True)
```

#### Example 3: Completed Run

```
State: Completion status exists (checkpoints may or may not exist)
Input: "single_bs16_cw128_lr1e-04"
Output: ("single_bs16_cw128_lr1e-04_2", False)
```

#### Example 4: Mixed Scenario

```
State:
- Base run: completed
- _2 run: completed
- _3 run: has checkpoints, not completed
Input: "single_bs16_cw128_lr1e-04"
Output: ("single_bs16_cw128_lr1e-04_3", True)
```

## Completion Tracking

### Completion Status Files

Located in `model_checkpoints/completion_tracking/<run_id>_completed.json`:

```json
{
  "run_id": "single_bs16_cw128_lr1e-04",
  "completed": true,
  "completion_timestamp": "2024-01-15T14:30:45.123456",
  "completion_reason": "training_completed_successfully",
  "final_step": 1000,
  "final_best_loss": 1.234,
  "best_checkpoint_logged": true,
  "post_training_eval_done": true
}
```

### Completion Reasons

- `"training_completed_successfully"`: Reached target steps
- `"early_stopping"`: Validation loss stopped improving
- `"exceeded_target_steps"`: Resumed run had already exceeded target steps
- `"user_interruption"`: Manual interruption (Ctrl+C, etc.)

### Lifecycle Integration

1. **Start**: Check completion status before initializing training
2. **During Training**: Save checkpoints periodically (no completion flags)
3. **On Completion**: Save completion metadata to JSON file
4. **After Upload**: Clean up local checkpoint files
5. **On Next Run**: Check completion status to decide resumption vs. new run

## Checkpoint Management

### Checkpoint Files

Two types of checkpoints are created during training:

- `run_{run_id}_bs{bs}_cw{cw}_lr{lr}_latest.pt`: Most recent model state
- `run_{run_id}_bs{bs}_cw{cw}_lr{lr}_best.pt`: Best validation loss model state

### Cleanup Strategy

After successful training completion and W&B artifact upload:

1. Upload best checkpoint as W&B artifact
2. Wait for upload confirmation (with timeout)
3. Delete local checkpoint files
4. Preserve completion status JSON

This ensures:

- No local storage waste from completed runs
- Completion status survives for future run ID decisions
- Interrupted runs keep their checkpoints for resumption

## W&B Integration

### Enhanced Metadata

All runs include metadata for better traceability:

```python
"run_metadata": {
  "created_timestamp": "2024-01-15T14:30:45.123456",
  "created_readable": "2024-01-15 14:30:45",
  "is_resuming": false,
  "python_version": "3.11.8"
}
```

### Run ID Conflicts

The system handles W&B run ID conflicts by:

1. Using deterministic base run IDs from hyperparameters
2. Adding incremental suffixes for new runs
3. Reusing the same ID for interrupted run resumption
4. Including timestamps in run metadata for unique identification

## Command Line Tools

### Managing Completed Runs

```bash
# List all completed runs
python scripts/manage_completed_runs.py list

# Check specific run status
python scripts/manage_completed_runs.py status "single_bs16_cw128_lr1e-04"

# Clear completion status (force re-run)
python scripts/manage_completed_runs.py clear "single_bs16_cw128_lr1e-04"
```

### Configuration

Smart resumption is automatically enabled and requires no configuration. It uses:

- `MODEL_CHECKPOINTS_DIR` for checkpoint and completion file storage
- Hyperparameter values to generate deterministic run IDs
- W&B project settings for artifact management

## Testing

### Unit Tests

Located in `tests/unit/training/test_smart_resumption.py`:

- Smart run ID generation scenarios
- Completion tracking integration
- Edge cases and error handling

### Integration Tests

Located in `tests/integration/training/test_training_lifecycle.py`:

- Full training lifecycle with resumption
- Completed run detection and skipping
- W&B integration with smart resumption

## Implementation Notes

### Thread Safety

The system is designed for single-process training but handles:

- Concurrent access to completion files (file locking)
- Race conditions in checkpoint detection
- W&B upload timeouts and failures

### Error Handling

Robust error handling for:

- Missing or corrupted completion files
- Failed checkpoint cleanup
- W&B API failures
- Filesystem permissions issues

### Performance

- Minimal overhead: O(1) completion checks
- Efficient suffix search: Early termination on first available
- Lazy checkpoint detection: Only when needed
- Batch completion file operations

## Migration Notes

### From Previous Systems

The current system replaces multiple overlapping completion tracking mechanisms:

- ✅ **JSON metadata files**: Single source of truth for completion status
- ❌ **Checkpoint-based completion**: Removed (conflicted with resumption)
- ❌ **In-memory completion flags**: Removed (not persistent)

### Backward Compatibility

- Existing checkpoints are automatically detected and handled
- Old completion tracking files are ignored (no migration needed)
- W&B runs continue to work with enhanced metadata

## Best Practices

### For Users

1. **Don't manually delete completion files** unless you want to force re-run
2. **Use the management script** to clear completion status safely
3. **Monitor W&B artifacts** to ensure uploads complete successfully
4. **Check logs** for resumption decisions and completion reasons

### For Developers

1. **Always check completion status** before starting training
2. **Use consistent run ID generation** for predictable behavior
3. **Handle W&B upload failures** gracefully (preserve local checkpoints)
4. **Test both fresh and resumed scenarios** in new features

## Troubleshooting

### Common Issues

**Q: Training skipped with "already completed" message**
A: Check completion status with `manage_completed_runs.py status <run_id>`, clear if needed

**Q: Unexpected run ID suffix (\_2, \_3, etc.)**
A: Previous runs with same hyperparameters were completed, check with `list` command

**Q: Resumption not working after interruption**
A: Verify checkpoint files exist and no completion status was accidentally created

**Q: W&B run ID conflicts**
A: Enable `resume="allow"` in W&B init (automatically handled by the system)

### Debug Mode

Enable detailed logging by setting log level to DEBUG:

```python
import logging
logging.getLogger("src.training.manager").setLevel(logging.DEBUG)
```

This provides detailed information about:

- Run ID generation decisions
- Completion status checks
- Checkpoint detection logic
- W&B integration steps

## Key Guarantees and Clarifications (as of latest version)

- **No Redundant Flags**: There are no in-memory or redundant completion flags in the codebase. The persistent JSON file in `model_checkpoints/completion_tracking/` is the _single source of truth_ for run completion status.
- **Config Path for Management Script**: The `manage_completed_runs.py` script always requires a config file path, defaulting to `configs/config.yml`. Use `--config` to specify a different path if needed.
- **Concurrent Run Safety**: The system is robust to multiple concurrent runs, as completion status and checkpoint detection are file-based and atomic.
- **Suffix Search Limit**: The smart run ID logic will search for available suffixes up to `_999`. If all are taken, it falls back to a timestamp-based run ID.
