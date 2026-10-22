import Foundation
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
        let modelVersion = ModelLocation.version(at: modelDirectory)
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
