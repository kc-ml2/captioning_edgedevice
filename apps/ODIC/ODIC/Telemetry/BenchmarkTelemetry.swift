import Foundation
import CryptoKit
import Darwin

/// Internal-test only. Bounded, atomic local JSON queue; no images or caption text.
actor BenchmarkTelemetry {
    static let shared = BenchmarkTelemetry()
    private let endpoint = TelemetryConfiguration.baseURL.appending(path: "v1/events/batch")
    private var worker: Task<Void, Never>?
    private var events: [BenchmarkEvent] = []
    private var loaded = false
    private let session: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 30
        configuration.timeoutIntervalForResource = 60
        return URLSession(configuration: configuration)
    }()

    private var queueURL: URL? {
        guard let directory = try? FileManager.default.url(for: .applicationSupportDirectory,
            in: .userDomainMask, appropriateFor: nil, create: true) else { return nil }
        let base = TelemetryConfiguration.baseURL.absoluteString
        if base == "https://scenesense.ml2-alpha.com" {
            return directory.appending(path: "benchmark-queue.json")
        }
        let suffix = SHA256.hash(data: Data(base.utf8)).map { String(format: "%02x", $0) }.joined()
        return directory.appending(path: "benchmark-queue-\(suffix).json")
    }

    func start() {
        guard TelemetryConfiguration.internalBenchmarkingEnabled,
              TelemetryConfiguration.ingestAPIKey != nil else { return }
        if !loaded {
            if let url = queueURL, let data = try? Data(contentsOf: url),
               let stored = try? JSONDecoder().decode([BenchmarkEvent].self, from: data) {
                events = Array(stored.suffix(500))
            }
            loaded = true
        }
        guard worker == nil, !events.isEmpty else { return }
        worker = Task { await self.drain() }
    }

    func record(_ event: BenchmarkEvent) {
        guard TelemetryConfiguration.internalBenchmarkingEnabled,
              TelemetryConfiguration.ingestAPIKey != nil else { return }
        start()
        events.append(event)
        events = Array(events.suffix(500))
        persist()
        start()
    }

    func recordFeedback(_ payload: CaptionFeedbackPayload, modelVersion: String) throws {
        guard TelemetryConfiguration.internalBenchmarkingEnabled,
              TelemetryConfiguration.ingestAPIKey != nil,
              let url = queueURL else { throw CocoaError(.fileWriteUnknown) }
        start()
        var event = BenchmarkEvent(modelVersion: modelVersion, eventType: "caption_feedback")
        event.caption_feedback = payload
        let updated = Array((events + [event]).suffix(500))
        let data = try JSONEncoder().encode(updated)
        try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
        events = updated
        start()
    }

    static func recordFailure(
        _ error: Error, kind: String, startedAt: Double,
        appState: String, downloadedBytes: Int64? = nil, stage: String? = nil
    ) async {
        guard TelemetryConfiguration.internalBenchmarkingEnabled,
              TelemetryConfiguration.ingestAPIKey != nil else { return }
        let nsError = error as NSError
        guard !(error is CancellationError),
              !(nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled) else { return }
        // Use a small allowlist, never arbitrary error descriptions, paths or URLs.
        let category: String
        switch nsError.domain {
        case NSURLErrorDomain: category = "network"
        case NSCocoaErrorDomain: category = "cocoa"
        case NSPOSIXErrorDomain: category = "posix"
        default: category = "application"
        }
        let version = ModelLocation.version(at: try? ModelLocation.installedDirectory())
        let details = FailureDetails(
            elapsed_ms: max(0, ProcessInfo.processInfo.systemUptime - startedAt) * 1000,
            error_code: nsError.code, error_category: category, app_state: appState,
            downloaded_bytes: downloadedBytes, stage: stage
        )
        var event = BenchmarkEvent(modelVersion: version, eventType: kind)
        if kind == "caption_failure" { event.caption_failure = details }
        else { event.download_failure = details }
        await shared.record(event)
    }

    private func persist() {
        guard var url = queueURL, let data = try? JSONEncoder().encode(events) else { return }
        do {
            try data.write(to: url, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
            var values = URLResourceValues()
            values.isExcludedFromBackup = true
            try url.setResourceValues(values)
        } catch {
            // Telemetry must never interrupt caption generation.
        }
    }

    private func drain() async {
        defer { worker = nil }
        var delay: UInt64 = 5
        while !events.isEmpty {
            let batch = Array(events.prefix(50))
            guard let key = TelemetryConfiguration.ingestAPIKey else { return }
            var request = URLRequest(url: endpoint)
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.setValue(key, forHTTPHeaderField: "X-Ingest-Key")
            do {
                request.httpBody = try JSONEncoder().encode(["events": batch])
                let (data, response) = try await session.data(for: request)
                guard let http = response as? HTTPURLResponse else { return }
                if http.statusCode == 200 {
                    struct Acknowledgement: Decodable { let acknowledged_event_ids: [UUID] }
                    let ack = try JSONDecoder().decode(Acknowledgement.self, from: data)
                    let sent = Set(batch.map(\.event_id))
                    let acknowledged = Set(ack.acknowledged_event_ids).intersection(sent)
                    guard !acknowledged.isEmpty else { return }
                    events.removeAll { acknowledged.contains($0.event_id) }
                    persist()
                    delay = 5
                    continue
                }
                // Invalid key/schema requires a developer fix, not an endless retry loop.
                if (400..<500).contains(http.statusCode), http.statusCode != 429 { return }
            } catch { }
            do { try await Task.sleep(nanoseconds: delay * 1_000_000_000) }
            catch { return }
            delay = min(delay * 2, 300)
        }
    }
}
