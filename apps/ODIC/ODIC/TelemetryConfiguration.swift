import Foundation

/// Internal-test ingestion credential. This is extractable from the app bundle,
/// not a user credential or proof that requests come from a genuine app.
nonisolated enum TelemetryConfiguration {
    // Explicit internal-build opt-in; missing local settings means no collection.
    static var internalBenchmarkingEnabled: Bool {
        guard let url = Bundle.main.url(forResource: "TelemetrySecrets", withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let values = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            return false
        }
        return values["InternalBenchmarkingEnabled"] as? Bool == true
    }

    static var ingestAPIKey: String? {
        guard let url = Bundle.main.url(forResource: "TelemetrySecrets", withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let values = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any],
              let key = values["IngestAPIKey"] as? String, key.count >= 32 else {
            return nil
        }
        return key
    }
}
