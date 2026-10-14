import Foundation
import CryptoKit
import Darwin

/// Samples process physical footprint. The sampled peak can miss short-lived spikes.
nonisolated final class BenchmarkMemorySampler: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private let timer: DispatchSourceTimer

    init() {
        timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "scenesense.memory"))
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
        sample()
    }

    private func sample() {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard status == KERN_SUCCESS else { return }
        lock.lock()
        peak = max(peak, info.phys_footprint)
        lock.unlock()
    }

    func stop() -> UInt64 {
        timer.cancel()
        sample()
        lock.lock()
        defer { lock.unlock() }
        return peak
    }

    deinit { timer.cancel() }
}

/// Per-caption measurements, owned by the pipeline's current invocation.
/// Loading is excluded from caption latency and memory sampling. EOS contributes
/// to token timing, but not to the generated-token count.
nonisolated struct CaptionBenchmarkSession {
    private let isColdRun: Bool
    private let loadStart: Double
    private var loadMilliseconds: Double = 0
    private var captionStart: Double = 0
    private var generationStart: Double = 0
    private var firstTokenTime: Double?
    private var lastTokenTime: Double = 0
    private var tokenCount = 0
    private var memorySampler: BenchmarkMemorySampler?

    init?(isColdRun: Bool) {
        guard TelemetryConfiguration.internalBenchmarkingEnabled else { return nil }
        self.isColdRun = isColdRun
        loadStart = ProcessInfo.processInfo.systemUptime
    }

    mutating func modelLoaded() {
        loadMilliseconds = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
        captionStart = ProcessInfo.processInfo.systemUptime
        memorySampler = BenchmarkMemorySampler()
    }

    mutating func generationStarted() {
        generationStart = ProcessInfo.processInfo.systemUptime
        lastTokenTime = generationStart
    }

    mutating func tokenGenerated(isEndOfSequence: Bool) {
        lastTokenTime = ProcessInfo.processInfo.systemUptime
        if firstTokenTime == nil { firstTokenTime = lastTokenTime }
        if !isEndOfSequence { tokenCount += 1 }
    }

    mutating func stop() {
        _ = stopMemorySampling()
    }

    mutating func record(outputWords: Int, modelDirectory: URL?) -> BenchmarkReference {
        let latency = (ProcessInfo.processInfo.systemUptime - captionStart) * 1000
        let generationSeconds = lastTokenTime - generationStart
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        var modelVersion = "unknown"
        if let directory = modelDirectory,
           let data = try? Data(contentsOf: directory.appending(path: "installed-manifest.json")),
           let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) {
            modelVersion = manifest.version
        }
        let event = BenchmarkEvent(metrics: CaptionBenchmark(
            is_cold_run: isColdRun, cold_load_ms: isColdRun ? loadMilliseconds : nil,
            caption_latency_ms: latency,
            time_to_first_token_ms: firstTokenTime.map { ($0 - captionStart) * 1000 },
            generation_ms: generationSeconds * 1000, generated_tokens: tokenCount,
            tokens_per_second: generationSeconds > 0 ? Double(tokenCount) / generationSeconds : nil,
            peak_sampled_memory_bytes: stopMemorySampling(),
            output_words: outputWords, thermal_state: thermal
        ), modelVersion: modelVersion)
        Task { await BenchmarkTelemetry.shared.record(event) }
        return BenchmarkReference(eventID: event.event_id, modelVersion: modelVersion)
    }

    private mutating func stopMemorySampling() -> UInt64 {
        let peak = memorySampler?.stop() ?? 0
        memorySampler = nil
        return peak
    }
}

nonisolated struct BenchmarkReference: Sendable {
    let eventID: UUID
    let modelVersion: String
}

nonisolated struct CaptionFeedbackPayload: Codable, Sendable {
    let benchmark_event_id: UUID
    let revision: Int
    let is_useful: Bool
    let reasons: [String]
}

nonisolated struct CaptionBenchmark: Codable, Sendable {
    let is_cold_run: Bool
    let cold_load_ms: Double?
    let caption_latency_ms: Double
    let time_to_first_token_ms: Double?
    let generation_ms: Double
    let generated_tokens: Int
    let tokens_per_second: Double?
    let peak_sampled_memory_bytes: UInt64
    // Optional fields preserve queued events from older app versions without
    // inventing a word count from a character count. New events emit words only.
    var output_words: Int? = nil
    var output_characters: Int? = nil
    let thermal_state: String
}

nonisolated struct FailureDetails: Codable, Sendable {
    let elapsed_ms: Double
    let error_code: Int
    let error_category: String
    let app_state: String
    let downloaded_bytes: Int64?
    let stage: String?
}

nonisolated struct BenchmarkEvent: Codable, Sendable {
    let schema_version: Int
    let event_id: UUID
    let event_type: String
    let timestamp: String
    let app_version: String
    let build_number: String
    let device_model: String
    let os_version: String
    let model_version: String
    var metrics: CaptionBenchmark?
    var caption_failure: FailureDetails?
    var download_failure: FailureDetails?
    var caption_feedback: CaptionFeedbackPayload?

    init(metrics: CaptionBenchmark? = nil, modelVersion: String, eventType: String = "caption_benchmark") {
        schema_version = 1
        event_type = eventType
        event_id = UUID()
        timestamp = ISO8601DateFormatter().string(from: Date())
        app_version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        build_number = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        var system = utsname()
        uname(&system)
        device_model = withUnsafeBytes(of: &system.machine) { bytes in
            String(decoding: bytes.prefix { $0 != 0 }, as: UTF8.self)
        }
        os_version = ProcessInfo.processInfo.operatingSystemVersionString
        model_version = modelVersion
        self.metrics = metrics
    }
}

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
        var version = "unknown"
        if let root = try? FileManager.default.url(for: .applicationSupportDirectory,
            in: .userDomainMask, appropriateFor: nil, create: false),
           let data = try? Data(contentsOf: root.appending(path: "MobileVLM/installed-manifest.json")),
           let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let value = manifest["version"] as? String {
            version = value
        }
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
