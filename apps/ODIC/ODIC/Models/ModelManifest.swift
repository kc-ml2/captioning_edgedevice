import Foundation

nonisolated struct ModelManifest: Codable, Sendable {
    struct File: Codable, Sendable {
        let path: String
        let size: Int64
        let sha256: String
    }

    let schemaVersion: Int
    let model: String
    let version: String
    let totalSize: Int64
    let files: [File]
}
