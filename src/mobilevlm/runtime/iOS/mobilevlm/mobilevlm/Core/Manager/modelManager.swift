import CoreML
import SentencepieceTokenizer

final class ModelManager {

    static let shared = ModelManager()

    // =========================
    // CoreML Models
    // =========================

    let visionModel: VisionEncoder_32
    let projectorModel: Projector_32
    let llmModel: mobilellama_32

    // =========================
    // Tokenizer
    // =========================

    let tokenizer: SentencepieceTokenizer

    // =========================
    // Embedding Weights
    // =========================

    let embeddingWeights: [Float]

    // =========================
    // Init
    // =========================

    private init() {

        let config = MLModelConfiguration()

        config.computeUnits = .all

        do {

            // =====================
            // Vision
            // =====================

            visionModel = try VisionEncoder_32(
                configuration: config
            )

            // =====================
            // Projector
            // =====================

            projectorModel = try Projector_32(
                configuration: config
            )

            // =====================
            // LLM
            // =====================

            llmModel = try mobilellama_32(
                configuration: config
            )

            // =====================
            // Tokenizer
            // =====================

            guard let tokenizerPath =
                Bundle.main.path(
                    forResource: "tokenizer",
                    ofType: "model"
                )
            else {

                fatalError(
                    "❌ tokenizer.model not found"
                )
            }

            tokenizer =
                try SentencepieceTokenizer(
                    modelPath: tokenizerPath
                )

            // =====================
            // Embedding Weights
            // =====================

            guard let embedURL =
                Bundle.main.url(
                    forResource: "embed_tokens",
                    withExtension: "bin"
                )
            else {

                fatalError(
                    "❌ embed_tokens.bin not found"
                )
            }

            let data =
                try Data(contentsOf: embedURL)

            embeddingWeights =
                data.withUnsafeBytes {

                    let buffer =
                        $0.bindMemory(
                            to: Float.self
                        )

                    return Array(buffer)
                }

            print("✅ All resources loaded")

        } catch {

            fatalError(
                "❌ Resource load failed: \(error)"
            )
        }
    }
}
