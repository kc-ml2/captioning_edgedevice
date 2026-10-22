import Foundation

nonisolated struct ModelConfiguration: Decodable {
    struct Quantization: Decodable {
        let bits: Int
        let group_size: Int
        let mode: String
    }
    let weight_dtype: String
    let quantization: Quantization
    static let bits = 4
    static let groupSize = 64

    static func validate(at directory: URL) throws {
        let config = try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appending(path: "config.json")))
        guard config.weight_dtype == "mixed_fp16_q4", config.quantization.bits == bits,
              config.quantization.group_size == groupSize, config.quantization.mode == "affine" else {
            throw ModelInstallError.unsupportedManifest
        }
    }
}

nonisolated enum ModelLocation {
    static func installedDirectory() throws -> URL {
        try FileManager.default.url(for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true).appending(path: "MobileVLM", directoryHint: .isDirectory)
    }

    static func version(at directory: URL?) -> String {
        guard let directory,
              let data = try? Data(contentsOf: directory.appending(path: "installed-manifest.json")),
              let manifest = try? JSONDecoder().decode(ModelManifest.self, from: data) else { return "unknown" }
        return manifest.version
    }
}
