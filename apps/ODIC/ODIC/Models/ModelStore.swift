import CryptoKit
import Foundation

nonisolated private struct LatestModel: Decodable {
    let schemaVersion: Int
    let version: String
    let manifestPath: String
}

nonisolated enum ModelInstallError: LocalizedError {
    case endpointMissing
    case invalidResponse
    case unsupportedManifest
    case unsafePath(String)
    case invalidFile(String)
    case insufficientStorage(required: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .endpointMissing:
            L10n.endpointMissing
        case .invalidResponse:
            L10n.invalidServerResponse
        case .unsupportedManifest:
            L10n.unsupportedManifest
        case .unsafePath(let path):
            L10n.unsafePath(path)
        case .invalidFile(let path):
            L10n.invalidFile(path)
        case .insufficientStorage(let required, let available):
            L10n.insufficientStorage(required: Self.size(required), available: Self.size(available))
        }
    }

    private static func size(_ value: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: value, countStyle: .file)
    }
}

actor ModelStore {
    struct Availability: Sendable {
        let version: String
        let totalSize: Int64

        var formattedSize: String {
            ByteCountFormatter.string(fromByteCount: totalSize, countStyle: .file)
        }
    }

    struct Progress: Sendable {
        let message: String
        let completedBytes: Int64
        let totalBytes: Int64

        var fraction: Double {
            guard totalBytes > 0 else { return 0 }
            return min(1, Double(completedBytes) / Double(totalBytes))
        }
    }

    private let fileManager = FileManager.default
    private let decoder = JSONDecoder()
    private let metadataSession: URLSession
    private let directoryOverride: URL?

    init(session: URLSession = .shared, directory: URL? = nil) {
        metadataSession = session
        directoryOverride = directory
    }

    var isInstalled: Bool {
        guard let directory = try? installedDirectory(),
              fileManager.fileExists(atPath: directory.appending(path: "installed-manifest.json").path),
              (try? ModelConfiguration.validate(at: directory)) != nil,
              let manifestData = try? Data(contentsOf: directory.appending(path: "installed-manifest.json")),
              let manifest = try? decoder.decode(ModelManifest.self, from: manifestData) else { return false }
        let paths = Set(manifest.files.map(\.path))
        guard paths.isSuperset(of: ["config.json", "tokenizer.model", "model.safetensors.index.json"]),
              let index = try? ModelWeightIndex.load(from: directory), !index.weightMap.isEmpty,
              Set(index.shards).isSubset(of: paths) else { return false }
        for file in manifest.files {
            guard let parts = try? safeRelativePath(file.path), file.size > 0,
                  let attributes = try? fileManager.attributesOfItem(atPath: parts.reduce(directory) { $0.appending(path: $1) }.path),
                  (attributes[.size] as? NSNumber)?.int64Value == file.size else { return false }
        }
        try? excludeFromBackup(directory)
        return true
    }

    func latestAvailability() async throws -> Availability {
        let (_, manifest) = try await latestManifest()
        return Availability(version: manifest.version, totalSize: manifest.totalSize)
    }

    func installLatest(progress: @escaping @Sendable (Progress) async -> Void) async throws {
        await progress(.init(message: L10n.checkingLatestModel, completedBytes: 0, totalBytes: 0))
        let (manifestURL, manifest) = try await latestManifest()

        let installed = try installedDirectory()
        if let current = try? Data(contentsOf: installed.appending(path: "installed-manifest.json")),
           let currentManifest = try? decoder.decode(ModelManifest.self, from: current),
           currentManifest.version == manifest.version, isInstalled {
            await progress(.init(message: L10n.modelReady, completedBytes: manifest.totalSize, totalBytes: manifest.totalSize))
            return
        }

        try ensureFreeSpace(for: manifest.totalSize, at: installed.deletingLastPathComponent())
        let staging = installed.deletingLastPathComponent().appending(
            path: ".MobileVLM-\(manifest.version)-download",
            directoryHint: .isDirectory
        )
        try? fileManager.removeItem(at: staging)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        try excludeFromBackup(staging)

        do {
            let releaseBase = manifestURL.deletingLastPathComponent()
            var completed: Int64 = 0
            for file in manifest.files {
                let relative = try safeRelativePath(file.path)
                let sourceURL = try relativeURL(file.path, under: releaseBase)
                let destination = relative.reduce(staging) { $0.appending(path: $1) }
                try fileManager.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)

                let filename = relative.last ?? file.path
                await progress(.init(
                    message: L10n.downloadingFile(filename),
                    completedBytes: completed,
                    totalBytes: manifest.totalSize
                ))
                let downloader = ModelFileDownloader(
                    source: sourceURL,
                    destination: destination,
                    completedBeforeFile: completed,
                    totalModelBytes: manifest.totalSize
                ) { downloaded, total in
                    await progress(.init(
                        message: L10n.downloadingFile(filename),
                        completedBytes: downloaded,
                        totalBytes: total
                    ))
                }
                try await downloader.start()
                try Task.checkCancellation()
                await progress(.init(
                    message: L10n.verifyingFile(filename),
                    completedBytes: completed + file.size,
                    totalBytes: manifest.totalSize
                ))
                try verify(destination, expectedSize: file.size, expectedSHA256: file.sha256, displayPath: file.path)
                completed += file.size
            }

            let manifestData = try JSONEncoder().encode(manifest)
            try manifestData.write(to: staging.appending(path: "installed-manifest.json"), options: .atomic)
            try activate(staging: staging, installed: installed)
            await progress(.init(message: L10n.installationComplete, completedBytes: manifest.totalSize, totalBytes: manifest.totalSize))
        } catch {
            try? fileManager.removeItem(at: staging)
            throw error
        }
    }

    private func latestManifest() async throws -> (URL, ModelManifest) {
        let baseURL = try distributionBaseURL()
        let latest: LatestModel = try await json(at: baseURL.appending(path: "latest.json"))
        guard latest.schemaVersion == 1 else { throw ModelInstallError.unsupportedManifest }
        let manifestURL = try relativeURL(latest.manifestPath, under: baseURL)
        let manifest: ModelManifest = try await json(at: manifestURL)
        guard manifest.schemaVersion == 1,
              manifest.version == latest.version,
              manifest.totalSize > 0,
              !manifest.files.isEmpty,
              manifest.files.allSatisfy({ $0.size > 0 && $0.size <= manifest.totalSize }),
              Set(manifest.files.map(\.path)).count == manifest.files.count,
              manifest.files.reduce(Int64(0), { sum, file in
                  let result = sum.addingReportingOverflow(file.size)
                  return result.overflow ? Int64.min : result.partialValue
              }) == manifest.totalSize else {
            throw ModelInstallError.unsupportedManifest
        }
        _ = try safeRelativePath(manifest.version)
        guard !manifest.version.contains("/") else { throw ModelInstallError.unsafePath(manifest.version) }
        for file in manifest.files { _ = try safeRelativePath(file.path) }
        return (manifestURL, manifest)
    }

    private func distributionBaseURL() throws -> URL {
        guard let value = Bundle.main.object(forInfoDictionaryKey: "ODICModelBaseURL") as? String,
              !value.isEmpty,
              let url = URL(string: value),
              url.scheme == "https" else {
            throw ModelInstallError.endpointMissing
        }
        return url
    }

    private func installedDirectory() throws -> URL {
        if let directoryOverride { return directoryOverride }
        return try ModelLocation.installedDirectory()
    }

    private func json<T: Decodable>(at url: URL) async throws -> T {
        let (data, response) = try await metadataSession.data(from: url)
        try validate(response)
        return try decoder.decode(T.self, from: data)
    }

    private func validate(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
            throw ModelInstallError.invalidResponse
        }
    }

    private func relativeURL(_ path: String, under base: URL) throws -> URL {
        let parts = try safeRelativePath(path)
        return parts.reduce(base) { $0.appending(path: $1) }
    }

    private func safeRelativePath(_ path: String) throws -> [String] {
        guard !path.hasPrefix("/"), !path.contains("\\") else {
            throw ModelInstallError.unsafePath(path)
        }
        let parts = path.split(separator: "/", omittingEmptySubsequences: false).map(String.init)
        guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw ModelInstallError.unsafePath(path)
        }
        return parts
    }

    private func verify(_ url: URL, expectedSize: Int64, expectedSHA256: String, displayPath: String) throws {
        let values = try url.resourceValues(forKeys: [.fileSizeKey])
        guard Int64(values.fileSize ?? -1) == expectedSize else {
            throw ModelInstallError.invalidFile(displayPath)
        }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            try Task.checkCancellation()
            let data = try handle.read(upToCount: 8 * 1024 * 1024) ?? Data()
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        let actual = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard actual.caseInsensitiveCompare(expectedSHA256) == .orderedSame else {
            throw ModelInstallError.invalidFile(displayPath)
        }
    }

    private func ensureFreeSpace(for modelSize: Int64, at url: URL) throws {
        let values = try url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let available = values.volumeAvailableCapacityForImportantUsage ?? 0
        // Download staging and an existing installation may coexist during an update.
        let required = modelSize + max(modelSize / 10, 256 * 1024 * 1024)
        guard available >= required else {
            throw ModelInstallError.insufficientStorage(required: required, available: available)
        }
    }

    private func excludeFromBackup(_ directory: URL) throws {
        var url = directory
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try url.setResourceValues(values)
    }

    func invalidateInstallation() throws {
        try fileManager.removeItem(at: installedDirectory().appending(path: "installed-manifest.json"))
    }

    private func activate(staging: URL, installed: URL) throws {
        let backup = installed.deletingLastPathComponent().appending(path: ".MobileVLM-previous")
        try? fileManager.removeItem(at: backup)
        if fileManager.fileExists(atPath: installed.path) {
            try fileManager.moveItem(at: installed, to: backup)
        }
        do {
            try fileManager.moveItem(at: staging, to: installed)
            try? fileManager.removeItem(at: backup)
        } catch {
            if fileManager.fileExists(atPath: backup.path) {
                try? fileManager.moveItem(at: backup, to: installed)
            }
            throw error
        }
    }
}
