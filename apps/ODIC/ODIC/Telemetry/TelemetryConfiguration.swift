import Foundation

/// Bundle credentials are extractable; they are not user authentication.
/// DEBUG selects the local environment independently of INTERNAL_TESTING features.
nonisolated enum TelemetryConfiguration {
    private static var values: [String: Any] {
        #if DEBUG
        let resource = "TelemetrySecrets-Debug"
        #else
        let resource = "TelemetrySecrets"
        #endif
        guard let url = Bundle.main.url(forResource: resource, withExtension: "plist"),
              let data = try? Data(contentsOf: url),
              let values = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] else {
            return [:]
        }
        return values
    }

    private static var configuredBaseURL: URL? {
        #if DEBUG
        guard let value = values["TelemetryBaseURL"] as? String,
              let url = URL(string: value),
              ["http", "https"].contains(url.scheme ?? ""),
              let host = url.host, !host.isEmpty,
              host != "scenesense.ml2-alpha.com",
              url.user == nil, url.password == nil,
              url.query == nil, url.fragment == nil else { return nil }
        return url
        #else
        return URL(string: "https://scenesense.ml2-alpha.com")!
        #endif
    }

    static var baseURL: URL {
        // Fail closed: missing debug configuration never falls back to production.
        configuredBaseURL ?? URL(string: "http://127.0.0.1:1")!
    }

    static var internalBenchmarkingEnabled: Bool {
        configuredBaseURL != nil && values["InternalBenchmarkingEnabled"] as? Bool == true
    }

    private static func key(_ name: String) -> String? {
        guard configuredBaseURL != nil,
              let key = values[name] as? String, key.count >= 32 else { return nil }
        return key
    }

    static var ingestAPIKey: String? { key("IngestAPIKey") }
}
