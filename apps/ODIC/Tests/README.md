# App regression checks

Run the Foundation-only checks on macOS (Xcode Swift toolchain):

```sh
swiftc -parse-as-library \
  apps/ODIC/ODIC/Speech/SpeechChunks.swift \
  apps/ODIC/ODIC/Models/ModelStore.swift \
  apps/ODIC/ODIC/Models/ModelFileDownloader.swift \
  apps/ODIC/ODIC/Models/ModelManifest.swift \
  apps/ODIC/ODIC/Models/ModelWeightIndex.swift \
  apps/ODIC/ODIC/Models/ModelConfiguration.swift \
  apps/ODIC/ODIC/Resources/Localization.swift \
  apps/ODIC/Tests/ReviewRegressionTests.swift -o /tmp/odic-review-tests
/tmp/odic-review-tests
```

Covers phrase boundaries/reselection after interruption, cancellation racing download
setup, and missing/truncated installed model files. This is not an iOS UI test suite.

Before distribution, verify on a physical iPhone:
- Silent mode + nonzero media volume: automatic speech without button interaction.
- Open/close Control Center mid-phrase: restart the interrupted phrase, then continue.
- Explicit Stop: no automatic restart on scene activation.
- Background during inference: cancel further GPU work; preserve generated partial text.
  Returning reads the interrupted phrase and remaining generated text, not ungenerated tokens.
- Background after complete caption: resume the interrupted phrase and rest of caption.
- Incoming call/audio interruption: recover without duplicate completed phrases.
- Repeated start/background/capture: no UI blocking or stuck capture continuation.
- Cancel a model download near completion, retry, and validate installation.
- Missing model shard: setup screen appears and re-download repairs installation.
- Internal review: empty comment, attachment, Unicode limit, offline retry, 401/409/422.

Generation timing retains the existing wall-clock definition, including UI callback waits.
Deploy the updated telemetry server to accept internal reviews with empty comments.
