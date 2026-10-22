#if INTERNAL_TESTING
import Foundation

/// A failed request retains its identity and exact body for idempotent retry.
struct InternalReviewDraft {
    let id = UUID()
    var body: Data?
    var isFrozen: Bool { body != nil }
}
#endif
