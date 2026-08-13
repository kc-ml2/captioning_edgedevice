//import Foundation
//
//final class EmbeddingManager {
//
//    static let shared = EmbeddingManager()
//
//    let embeddingWeights: [Float]
//
//    private init() {
//
//        guard let url = Bundle.main.url(
//            forResource: "embed_tokens",
//            withExtension: "bin"
//        ) else {
//
//            fatalError("embed_tokens.bin not found")
//        }
//
//        guard let data = try? Data(contentsOf: url) else {
//
//            fatalError("Failed to read embed_tokens.bin")
//        }
//
//        self.embeddingWeights = data.withUnsafeBytes {
//
//            let buffer = $0.bindMemory(to: Float.self)
//
//            return Array(buffer)
//        }
//
//        print("✅ Embedding weights loaded once")
//    }
//}
