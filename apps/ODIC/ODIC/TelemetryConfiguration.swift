import Foundation

/// Internal-test ingestion credential. This is extractable from the app bundle,
/// not a user credential or proof that requests come from a genuine app.
nonisolated enum TelemetryConfiguration {
    static var ingestAPIKey: String? {
        guard let url = Bundle.main.url(forResource: "TelemetrySecrets", withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let values = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: String],
              let key = values["IngestAPIKey"], key.count >= 32 else {
            return nil
        }
        return key
    }
}
