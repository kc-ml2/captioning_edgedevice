import CoreML
import SentencepieceTokenizer

final class ModelManager {

    static let shared = ModelManager()

    let visionModel: vit_fp16
    let projectorModel: projector_fp16
    let embeddingModel: embeddinglayer_fp16_cp16
    let lmheadModel: lm_head_fp16
    
    let llmModelURL: URL


    // =========================
    // Tokenizer
    // =========================

    let tokenizer: SentencepieceTokenizer

    private init() {

        let config = MLModelConfiguration()
        config.computeUnits = .all

        do {

            visionModel = try vit_fp16(configuration: config)
            projectorModel = try projector_fp16(configuration: config)
            embeddingModel = try embeddinglayer_fp16_cp16(configuration: config)
            lmheadModel = try lm_head_fp16(configuration: config)

            // =====================================
            // LLM
            // =====================================
            
            guard let llmURL =
                Bundle.main.url(
                    forResource: "mobilellama_multifunction_fp16",
                    withExtension: "mlmodelc"
                )
            else {
                fatalError("LLM model not found")
            }
            
            llmModelURL = llmURL
            
            // =====================================
            // Tokenizer
            // =====================================

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

        } catch {

            fatalError(
                "❌ Resource load failed: \(error)"
            )
        }
    }
}
