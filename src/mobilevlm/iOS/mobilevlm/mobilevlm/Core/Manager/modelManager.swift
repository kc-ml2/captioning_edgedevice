import CoreML
import SentencepieceTokenizer

final class ModelManager {

    static let shared = ModelManager()

    // =========================
    // CoreML Models
    // =========================

    let visionModel: VisionEncoder_int8
    let projectorModel: Projector_int8
    let llmModel: mobilellama_int8

    // =========================
    // Tokenizer
    // =========================

    let tokenizer: SentencepieceTokenizer

    // =========================
    // Embedding Weights
    // =========================

    let embeddingWeights: [Int8]

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

            visionModel = try VisionEncoder_int8(
                configuration: config
            )

            // =====================
            // Projector
            // =====================

            projectorModel = try Projector_int8(
                configuration: config
            )

            // =====================
            // LLM
            // =====================

            llmModel = try mobilellama_int8(
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
                    forResource: "embed_tokens_int8",
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
                            to: Int8.self
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
