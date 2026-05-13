// import Foundation

// func loadEmbeddingWeights() -> [Float] {

//     guard let url = Bundle.main.url(
//         forResource: "embed_tokens",
//         withExtension: "bin"
//     ) else {
//         fatalError("embed_tokens.bin not found")
//     }

//     guard let data = try? Data(contentsOf: url) else {
//         fatalError("Failed to read embed_tokens.bin")
//     }

//     let weights: [Float] = data.withUnsafeBytes { rawBuffer in

//         let buffer = rawBuffer.bindMemory(to: Float.self)

//         return Array(buffer)
//     }

//     return weights
// }
