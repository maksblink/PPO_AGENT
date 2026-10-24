# Cold storage — latest verified backup only

The laptop project is authoritative. The external disk holds one complete,
verified generation after each successful `sync`. The tool never empties a disk
or edits sibling folders. It manages only `PPO_AGENT_backup_<project UUID>` at
the selected mount point. The UUID comes from `.cold_storage/identity.json`;
retain that file. Other files on the external disk are outside the backup scope.

## Commands

Run from the project root with its virtual environment active:

```bash
python extra_tools/maintenance/cold_storage.py disks
python extra_tools/maintenance/cold_storage.py probe --disk /run/media/maks/Expansion
python extra_tools/maintenance/cold_storage.py sync --disk /run/media/maks/Expansion
python extra_tools/maintenance/cold_storage.py verify --disk /run/media/maks/Expansion
```

| Mode | Behavior |
| --- | --- |
| `disks` | Lists devices and mounted destinations. |
| `probe` | Tests writing, reading and restoring temporary sample data; removes only its own temporary directories. |
| `scan` | Hashes all project files and stores a local PostgreSQL dump and inventory. This writes local state, not an external backup. |
| `compare` | Compares the latest saved local inventory with the disk backup. It does not rescan live files; run `scan` first when needed. |
| `sync` | Captures the project and database, verifies content, publishes the latest backup, then automatically removes older generations and unreferenced objects. |
| `verify` | Verifies current manifest integrity and SHA256 of all referenced content, including database dumps. It is not a PostgreSQL restore test. |
| `prune` | Verifies the current backup and explicitly finishes/removes older generations, incomplete generations and unreferenced objects. Requires confirmation when deletions are needed. |
| `restore-files` | Reconstructs verified files into a new, nonexistent directory. |
| `restore-database` | Restores the captured application database only when the target database does not exist. |

## Synchronization and retention

1. Acquire project/destination locks and require clean Git and stopped writers.
2. Verify the existing current backup, if any. Finish a marked interrupted cleanup.
3. Scan the laptop; compare paths, content hashes and recorded metadata.
4. Request one confirmation covering backup creation and latest-only retention.
5. Dump PostgreSQL in full. Copy complete new/modified content into SHA256 objects.
   Unchanged content is verified and reused; it is not copied again.
6. Verify all required objects and recheck source stability.
7. Atomically publish `CURRENT` only after the new backup is verified.
8. Delete older generation manifests and objects not referenced by the latest one.

A removed laptop file is absent from the latest manifest. Its old content is
removed unless another current file has identical bytes. A modified file is
stored in full; there are no byte-level delta patches. Unchanged contents can
be reused even if their path was renamed. Permissions, ownership, timestamps,
symlinks and extended attributes are represented in the manifest.

The previous complete backup remains available during copying. A failed copy,
dump, verification or pre-publication source-stability check does not replace it.
After publication, cleanup failure leaves the new backup usable and a durable
`cleanup_pending.json` marker. The next `sync` completes that cleanup before
starting another capture; `prune` can also finish it. During an interrupted
operation, extra generations/objects may temporarily remain. Successful `sync`
leaves only the latest generation and its referenced payload.

The tool uses backup format 2; the retention update does not change that format.
It does not automatically convert format-1 external archives. A successful sync
normalizes the local identity record to format 2 while preserving its UUID.
Older local `scan` capture folders are local working snapshots, not external
backup generations; this update does not automatically delete those captures.

## Scope and safety

All project files are included regardless of Git ignore rules: source, `.git`,
configs, data, artifacts, `.env`, and `.venv`. `.cold_storage` control state is
excluded from the file payload. PostgreSQL is captured separately along with
cluster globals in `roles.sql`. The archive contains credentials/secrets from
these sources and is not encrypted by this tool.

External symlink targets, including an interpreter in `/opt`, are not copied.
The virtual environment may require recreation on a different system/path.

Live capture and database restoration exclude guarded queues, rename and cleanup
operations. Direct/manual writers must also remain stopped. The tool does not
change CPU, GPU, fan or thermal settings. All external deletions are scoped to
the verified, exclusively managed backup folder; other disk folders are untouched.

## Restoration

`RESTORE.py` is copied to the backup folder and can restore files standalone:

```bash
python /path/to/backup/RESTORE.py restore-files \
  --backup /path/to/backup --target /path/to/new/project
```

Restore files to a Linux filesystem that supports the recorded metadata. Then
prepare the matching PostgreSQL major version and required roles, configure the
restored project's connection, and restore the database:

```bash
python /path/to/backup/RESTORE.py restore-database --project /path/to/new/project
```

The database must not already exist. `roles.sql` is saved but is not automatically
applied. Live modes load maintenance guards from the chosen project; an older
restored project must have those tools installed first. Verify the restored
project/database and environment before training. A failed database restore can
leave a partial database; it is never silently overwritten on retry.
