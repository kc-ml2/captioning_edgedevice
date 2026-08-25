import CryptoKit
import Foundation

nonisolated struct ModelManifest: Decodable, Sendable {
    struct File: Decodable, Sendable {
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

nonisolated private struct LatestModel: Decodable {
    let schemaVersion: Int
    let version: String
    let manifestPath: String
}

enum ModelInstallError: LocalizedError {
    case endpointMissing
    case invalidResponse
    case unsupportedManifest
    case unsafePath(String)
    case invalidFile(String)
    case insufficientStorage(required: Int64, available: Int64)

    var errorDescription: String? {
        switch self {
        case .endpointMissing:
            "모델 다운로드 주소가 설정되지 않았어요."
        case .invalidResponse:
            "모델 서버의 응답이 올바르지 않아요."
        case .unsupportedManifest:
            "지원하지 않는 모델 manifest예요."
        case .unsafePath(let path):
            "안전하지 않은 모델 파일 경로예요: \(path)"
        case .invalidFile(let path):
            "다운로드한 모델 파일이 손상되었어요: \(path)"
        case .insufficientStorage(let required, let available):
            "저장 공간이 부족해요. 필요: \(Self.size(required)), 사용 가능: \(Self.size(available))"
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

    init(session: URLSession = .shared) {
        metadataSession = session
    }

    var isInstalled: Bool {
        guard let directory = try? installedDirectory(),
              fileManager.fileExists(atPath: directory.appending(path: "installed-manifest.json").path),
              let data = try? Data(contentsOf: directory.appending(path: "config.json")),
              let config = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              config["weight_dtype"] as? String == "mixed_fp16_q4",
              let quantization = config["quantization"] as? [String: Any],
              quantization["bits"] as? Int == 4,
              quantization["group_size"] as? Int == 64 else {
            return false
        }
        return true
    }

    func latestAvailability() async throws -> Availability {
        let (_, manifest) = try await latestManifest()
        return Availability(version: manifest.version, totalSize: manifest.totalSize)
    }

    func installLatest(progress: @escaping @Sendable (Progress) async -> Void) async throws {
        await progress(.init(message: "최신 모델을 확인하고 있어요…", completedBytes: 0, totalBytes: 0))
        let (manifestURL, manifest) = try await latestManifest()

        let installed = try installedDirectory()
        if let current = try? Data(contentsOf: installed.appending(path: "installed-manifest.json")),
           let currentManifest = try? decoder.decode(ModelManifest.self, from: current),
           currentManifest.version == manifest.version {
            await progress(.init(message: "모델이 준비되었어요.", completedBytes: manifest.totalSize, totalBytes: manifest.totalSize))
            return
        }

        try ensureFreeSpace(for: manifest.totalSize, at: installed.deletingLastPathComponent())
        let staging = installed.deletingLastPathComponent().appending(
            path: ".MobileVLM-\(manifest.version)-download",
            directoryHint: .isDirectory
        )
        try? fileManager.removeItem(at: staging)
        try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)

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
                    message: "모델을 받고 있어요: \(filename)",
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
                        message: "모델을 받고 있어요: \(filename)",
                        completedBytes: downloaded,
                        totalBytes: total
                    ))
                }
                try await downloader.start()
                try Task.checkCancellation()
                await progress(.init(
                    message: "파일을 확인하고 있어요: \(filename)",
                    completedBytes: completed + file.size,
                    totalBytes: manifest.totalSize
                ))
                try verify(destination, expectedSize: file.size, expectedSHA256: file.sha256, displayPath: file.path)
                completed += file.size
            }

            let manifestData = try JSONEncoder().encode(EncodableManifest(manifest))
            try manifestData.write(to: staging.appending(path: "installed-manifest.json"), options: .atomic)
            try activate(staging: staging, installed: installed)
            await progress(.init(message: "모델 설치가 완료됐어요.", completedBytes: manifest.totalSize, totalBytes: manifest.totalSize))
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
              manifest.files.reduce(Int64(0), { $0 + $1.size }) == manifest.totalSize else {
            throw ModelInstallError.unsupportedManifest
        }
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
        try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appending(path: "MobileVLM", directoryHint: .isDirectory)
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

private final class ModelFileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    typealias ProgressHandler = @Sendable (Int64, Int64) async -> Void

    private let source: URL
    private let destination: URL
    private let completedBeforeFile: Int64
    private let totalModelBytes: Int64
    private let progress: ProgressHandler
    private var continuation: CheckedContinuation<Void, Error>?
    private var session: URLSession?
    private var task: URLSessionDownloadTask?
    private var result: Result<Void, Error>?

    init(
        source: URL,
        destination: URL,
        completedBeforeFile: Int64,
        totalModelBytes: Int64,
        progress: @escaping ProgressHandler
    ) {
        self.source = source
        self.destination = destination
        self.completedBeforeFile = completedBeforeFile
        self.totalModelBytes = totalModelBytes
        self.progress = progress
    }

    func start() async throws {
        try Task.checkCancellation()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                self.continuation = continuation
                if let result {
                    self.continuation = nil
                    continuation.resume(with: result)
                    return
                }
                let configuration = URLSessionConfiguration.default
                configuration.waitsForConnectivity = true
                configuration.allowsConstrainedNetworkAccess = true
                configuration.timeoutIntervalForResource = 60 * 60 * 6
                let session = URLSession(configuration: configuration, delegate: self, delegateQueue: nil)
                self.session = session
                let task = session.downloadTask(with: source)
                self.task = task
                task.resume()
            }
        } onCancel: {
            self.cancel()
        }
    }

    private func cancel() {
        task?.cancel()
        finish(.failure(CancellationError()))
    }

    private func finish(_ result: Result<Void, Error>) {
        guard self.result == nil else { return }
        self.result = result
        task = nil
        session?.finishTasksAndInvalidate()
        session = nil
        guard let continuation else { return }
        self.continuation = nil
        continuation.resume(with: result)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        let completed = min(totalModelBytes, completedBeforeFile + totalBytesWritten)
        Task { await progress(completed, totalModelBytes) }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse,
                  (200..<300).contains(response.statusCode) else {
                throw ModelInstallError.invalidResponse
            }
            let fileManager = FileManager.default
            try? fileManager.removeItem(at: destination)
            try fileManager.moveItem(at: location, to: destination)
            finish(.success(()))
        } catch {
            finish(.failure(error))
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            finish(.failure(error))
        }
    }
}

// Keeps the wire manifest Decodable-only while allowing the installed copy to be encoded.
nonisolated private struct EncodableManifest: Encodable {
    let schemaVersion: Int
    let model: String
    let version: String
    let totalSize: Int64
    let files: [File]

    struct File: Encodable {
        let path: String
        let size: Int64
        let sha256: String
    }

    init(_ manifest: ModelManifest) {
        schemaVersion = manifest.schemaVersion
        model = manifest.model
        version = manifest.version
        totalSize = manifest.totalSize
        files = manifest.files.map { .init(path: $0.path, size: $0.size, sha256: $0.sha256) }
    }
}
