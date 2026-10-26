# Cold storage — latest verified backup only

The laptop project is authoritative. The external disk holds one complete,
verified generation after each successful `sync`. The tool never empties a disk
or edits sibling folders. It manages only `PPO_AGENT_backup_<project UUID>` at
the selected mount point. The UUID comes from `.cold_storage/identity.json`;
retain that file. Other files on the external disk are outside the backup scope.

## Commands

Run from the project root with its virtual environment active. Replace
`/path/to/mounted/disk` with the actual mount point, not a subdirectory:

```bash
python extra_tools/maintenance/cold_storage.py disks
python extra_tools/maintenance/cold_storage.py probe --disk /path/to/mounted/disk
python extra_tools/maintenance/cold_storage.py sync --disk /path/to/mounted/disk
python extra_tools/maintenance/cold_storage.py verify --disk /path/to/mounted/disk
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

Each capture creates a fresh full database dump and a globals/roles dump. Their
bytes can change even if application rows do not. Identical dump bytes reuse an
existing object; different bytes create a new object. Obsolete dump objects are
removed after publication, unless still referenced by the current manifest.
Only the latest captured database state remains after successful cleanup.

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

Capture requires a running PostgreSQL server and matching-major `pg_dump`, plus
`pg_dumpall` and `pg_restore`. `DATABASE_URL` or `.env` supplies the connection.
The database account must be able to lock tables, export snapshots and dump globals.
The capture context holds source table write locks through confirmation, transfer
and finalization; do not leave a capture waiting for input while expecting writers
to proceed. A standalone clean Git checkout is required; external Git object
stores, submodules and unsupported special files are rejected.

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

The connection must use the original application database name on a server with
the same PostgreSQL major version as the capture. For a rehearsal, use a separate
PostgreSQL instance initialized with `POSTGRES_DB=postgres`; do not connect to the
production instance. The database must not already exist. `roles.sql` is saved but is not automatically
applied. Live modes load maintenance guards from the chosen project; an older
restored project must have those tools installed first. Verify the restored
project/database and environment before training. A failed database restore can
leave a partial database; it is never silently overwritten on retry.


## Local working state

The external latest-only policy does not remove local `scan` captures.

| Local path under `.cold_storage/` | Purpose |
| --- | --- |
| `identity.json` | Stable project UUID used to find the managed backup folder. |
| `lock` | Serializes local cold-storage operations; never unlink it during an operation. |
| `plan.json` | Inventory and proposed changes, marked `PLANNED_NOT_BACKED_UP`. |
| `manifest.json` | Last successfully captured local manifest. |
| `capture.json`, `capture-*` | Pointer and locally retained database captures made by `scan`. |
| `last_destination.json` | Last committed external backup location and generation. |
| `comparison.json` | Result of the most recent manifest comparison. |
| `last_operation_summary.json` | Timing/I/O summary, overwritten by the next command. |

Not every command creates every file. Even `disks` and `probe` initialize local
state and may replace the summary. Preserve the identity for future updates to
the same backup. Resetting it creates a new project UUID, not an update to the
old backup folder. Do not remove working state during an active operation.

## Progress and performance

`N/M/D` describe new, modified and deleted paths, including directory/link and
metadata changes. They do not count database rows. `New file content bytes`
excludes the not-yet-created database dump. `copied` measures newly stored object
bytes; `reused objects` counts existing content used without copying it again.
Multiple paths with identical content share one object.

`GiB I/O` includes reads and writes and can greatly exceed backup size. With an
existing backup, the current implementation verifies the old generation, checks
reused objects during storage, verifies the proposed generation, and verifies the
current generation again before pruning. Source files are also hashed twice.
These repeated reads are implemented behavior, not a promised optimization.

The total timer includes waiting for `YES`; that wait appears in
`setup/confirmation/finalization` along with unclassified work. Do not interpret
that entire remainder as disk processing time. A successful `sync` already
includes content verification; a separate `verify` can be used for a later check.

## Validation status and remaining rehearsal

Practical checks reported on 24–25 October 2026:

- Destination `probe`: passed the temporary file/metadata round-trip checks.
- Initial full backup: passed; about 199.97 GiB stored in 129,275 new objects.
- Repeat backup: `N=0, M=0, D=0`; 129,273 objects reused, two new objects totalling
  about 0.38 GiB copied. The old generation and two obsolete objects were removed.
- The project regression suite passed 850 tests before these practical checks.

These results validate the observed captures and latest-only cleanup. They do
not establish a complete production restore. Still to perform: controlled
add/modify/delete tests using disposable files, a complete file restore into a
new directory, and a database restore into an isolated PostgreSQL instance with
schema, row/sequence and artifact-reference checks. Do not delete source data to
conduct a rehearsal. A checksum check alone cannot prove the restored application
will run with the target system's interpreter, drivers and dependencies.
