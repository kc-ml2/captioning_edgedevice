import Foundation
import Darwin

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
