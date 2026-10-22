import Foundation

@main
struct ReviewRegressionTests {
    static func main() async throws {
        assert(SpeechChunks.next(text: "A chair. A table.", completedPrefix: "", final: true) == "A chair.")
        assert(SpeechChunks.next(text: "A chair. A table.", completedPrefix: "A chair.", final: true) == " A table.")
        assert(SpeechChunks.next(text: "A chair", completedPrefix: "", final: false) == nil)
        assert(SpeechChunks.next(text: "one two three four five six seven eight nine", completedPrefix: "", final: false) == "one two three four five six seven eight")
        assert(SpeechChunks.next(text: "Revised", completedPrefix: "Old", final: true) == nil)
        // An interrupted phrase leaves the prefix unchanged and is selected again.
        let text = "A chair. A table."
        let interrupted = SpeechChunks.next(text: text, completedPrefix: "A chair.", final: true)
        assert(interrupted == SpeechChunks.next(text: text, completedPrefix: "A chair.", final: true))

        // Race cancellation against downloader setup/completion repeatedly.
        let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        for index in 0..<100 {
            let downloader = ModelFileDownloader(source: URL(string: "http://127.0.0.1:1/test")!,
                destination: root.appending(path: "\(index)"), completedBeforeFile: 0, totalModelBytes: 1) { _, _ in }
            let task = Task { try await downloader.start() }
            if index % 2 == 0 { await Task.yield() }
            task.cancel()
            do { try await task.value; assertionFailure("Canceled download succeeded") } catch { }
        }
        let files: [String: Data] = [
            "config.json": Data(#"{"weight_dtype":"mixed_fp16_q4","quantization":{"bits":4,"group_size":64,"mode":"affine"}}"#.utf8),
            "model.safetensors.index.json": Data(#"{"weight_map":{"weight":"weights.safetensors"}}"#.utf8),
            "weights.safetensors": Data([1, 2, 3]),
            "tokenizer.model": Data([4, 5])
        ]
        for (name, data) in files { try data.write(to: root.appending(path: name)) }
        let manifest = ModelManifest(schemaVersion: 1, model: "test", version: "1", totalSize: Int64(files.values.reduce(0) { $0 + $1.count }),
            files: files.map { ModelManifest.File(path: $0.key, size: Int64($0.value.count), sha256: "unused") })
        try JSONEncoder().encode(manifest).write(to: root.appending(path: "installed-manifest.json"))
        let store = ModelStore(directory: root)
        let valid = await store.isInstalled
        assert(valid)
        try FileManager.default.removeItem(at: root.appending(path: "weights.safetensors"))
        let missing = await store.isInstalled
        assert(!missing)
        try Data([0]).write(to: root.appending(path: "weights.safetensors"))
        let truncated = await store.isInstalled
        assert(!truncated)
        print("Speech chunk, download cancellation, and damaged-install regression tests passed")
    }
}
