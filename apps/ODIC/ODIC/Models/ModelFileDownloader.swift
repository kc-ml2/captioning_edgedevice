import Foundation

nonisolated final class ModelFileDownloader: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    typealias ProgressHandler = @Sendable (Int64, Int64) async -> Void

    private let source: URL
    private let destination: URL
    private let completedBeforeFile: Int64
    private let totalModelBytes: Int64
    private let progress: ProgressHandler
    // Recursive because completion helpers are also called inside file-move sections.
    private let lock = NSRecursiveLock()
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
                lock.lock()
                defer { lock.unlock() }
                self.continuation = continuation
                if let result {
                    self.continuation = nil
                    continuation.resume(with: result)
                    return
                }
                let configuration = URLSessionConfiguration.default
                configuration.waitsForConnectivity = true
                configuration.allowsConstrainedNetworkAccess = true
                // Allow longer stalls on slow networks; this is separate from the total download limit.
                configuration.timeoutIntervalForRequest = 60 * 5
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
        lock.lock()
        defer { lock.unlock() }
        task?.cancel()
        finish(.failure(CancellationError()))
    }

    private func finish(_ result: Result<Void, Error>) {
        lock.lock()
        defer { lock.unlock() }
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
        lock.lock()
        defer { lock.unlock() }
        guard result == nil else { return }
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
