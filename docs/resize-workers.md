# Native image resizing

The editor launches its own executable with an internal resize-worker flag.
This is not an optional download and does not add another distributed executable.
The original resampling and pixel-art algorithms are unchanged.

The UI retains a document snapshot and immutable tiled pixel references. A
background task writes those pixels in bounded strips into a private job folder,
then starts the worker. Layers are processed sequentially and the worker uses at
most two Rayon threads. The result is checked for the exact expected byte count
and rebuilt into tiles off the UI thread. Only a successful, current document
result reaches undo history.

Cancel is checked during preparation, worker execution and result loading. The
parent terminates and waits for the disposable process rather than terminating
an in-process thread. A pipe owned by the parent also stops a worker if its
editor exits. Graceful editor shutdown requests cancellation and allows up to
two seconds for background cleanup. Temporary job files are removed on normal
completion, cancellation and reported errors. An abrupt OS/process termination
can leave a job folder in the OS temporary directory.

Worker requests carry a protocol marker, dimensions, bounded JSON options and
an optional protection mask. Their exact size and dimensions are validated
before pixel allocation. The worker uses the existing bounded-process memory
policy and has a ten-minute parent timeout. A codec or memory failure cannot
replace the original document. Result application still checks document identity,
revision and cancellation.

Navigation uses one Fit action for toolbar, menu and keyboard. Fitting leaves a
12-point margin on each side. Zoom-out stops at a tenth of the fitted size,
capped at 100% for small images; the absolute numerical floor remains 0.01%.
Fit and 100% are distinct actions.

Regression coverage is in `tests/resize_worker.rs`, `tests/processing_limits.rs`
and `src/app/followup_tests.rs`. Native tests compare each interpolation and
retarget mode, cancel a live resample, then run another job. UI tests activate
the actual toolbar/menu controls and shortcut in both light and dark themes.
