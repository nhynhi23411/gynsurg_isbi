# Local resume addendum — 19 October 2026

After167 accepted receipts, Modal reported the owned batch app already stopped but returned exit status1. The frozen controller therefore exited before retrying three hardware-allocation failures. No failed-quality case is retried.

`scripts/resume_train_expansion_recovery.py` invokes the unchanged frozen driver with one local subprocess adapter. Only `python -m modal app stop ap-... --yes` receives the adapter. An explicit `App is already stopped.` response is accepted as terminal cleanup; authentication, network and other errors still raise. Replies are recorded. Existing call/attempt/app ledgers are copied into timestamped resume history before the frozen controller writes its current ledger.

Scientific source, worker, local analysis environment contract, inputs, models, query trajectory, seed, A10 requirement and scoring remain unchanged. Resume discovers completed receipts from private volume; accepted inference is not repeated. Addendum and adapter/test hashes are recorded before resume in IDEMPOTENT_STOP_FREEZE.json.
