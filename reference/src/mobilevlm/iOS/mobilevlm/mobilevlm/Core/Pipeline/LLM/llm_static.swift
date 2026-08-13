import CoreML


//func runsingleLLM(
//    inputsEmbeds: MLMultiArray,
//    attentionMask: MLMultiArray,
//    kvCache: MLMultiArray
//) -> (
//    hidden_states: MLMultiArray,
//    presentKV: MLMultiArray
//)? {
//
//    do {
//
//        let LLMmodel = ModelManager.shared.singleLLMModel
//
//        let provider =
//            try MLDictionaryFeatureProvider(
//                dictionary: [
//                    "inputs_embeds": inputsEmbeds,
//                    "attention_mask": attentionMask,
//                    "kv_cache": kvCache
//                ]
//            )
//
//        let prediction =
//            try LLMmodel.model.prediction(
//                from: provider
//            )
//
//        guard
//            let hidden_states =
//                prediction.featureValue(
//                    for: "hidden_states"
//                )?.multiArrayValue,
//
//            let presentKV =
//                prediction.featureValue(
//                    for: "present_kv"
//                )?.multiArrayValue
//        else {
//
//            return nil
//        }
//
//        return (
//            hidden_states: hidden_states,
//            presentKV: presentKV
//        )
//
//    } catch {
//
//        print(error)
//        return nil
//    }
//}


import CoreML

enum LLMMode: String {
    case prefill
    case decode
}

final class LLMManager {

    static let shared = LLMManager()

    private init() {}

    private var prefillModel: MLModel?
    private var decodeModel: MLModel?

    // =====================================================
    // Load
    // =====================================================

    func loadPrefill(
        modelURL: URL
    ) throws {

        guard prefillModel == nil else {
            return
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all
        config.functionName = LLMMode.prefill.rawValue

        prefillModel = try MLModel(
            contentsOf: modelURL,
            configuration: config
        )
    }

    func loadDecode(
        modelURL: URL
    ) throws {

        guard decodeModel == nil else {
            return
        }

        let config = MLModelConfiguration()
        config.computeUnits = .all
        config.functionName = LLMMode.decode.rawValue

        decodeModel = try MLModel(
            contentsOf: modelURL,
            configuration: config
        )
    }

    // =====================================================
    // Unload
    // =====================================================

    func unloadPrefill() {
        prefillModel = nil
    }

    func unloadDecode() {
        decodeModel = nil
    }

    // =====================================================
    // Unified Inference
    // =====================================================

    func run(
        mode: LLMMode,
        inputsEmbeds: MLMultiArray,
        attentionMask: MLMultiArray,
        kvCache: MLMultiArray
    ) throws -> (
        hidden_states: MLMultiArray,
        presentKV: MLMultiArray
    ) {

        let model: MLModel

        switch mode {

        case .prefill:

            guard let prefillModel else {
                throw NSError(
                    domain: "LLMManager",
                    code: -1,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Prefill model is not loaded"
                    ]
                )
            }

            model = prefillModel

        case .decode:

            guard let decodeModel else {
                throw NSError(
                    domain: "LLMManager",
                    code: -2,
                    userInfo: [
                        NSLocalizedDescriptionKey:
                            "Decode model is not loaded"
                    ]
                )
            }

            model = decodeModel
        }

        let input = try MLDictionaryFeatureProvider(
            dictionary: [
                "inputs_embeds":
                    MLFeatureValue(
                        multiArray: inputsEmbeds
                    ),

                "attention_mask":
                    MLFeatureValue(
                        multiArray: attentionMask
                    ),

                "kv_cache":
                    MLFeatureValue(
                        multiArray: kvCache
                    )
            ]
        )

        let output = try model.prediction(
            from: input
        )

        guard
            let hiddenStates =
                output.featureValue(
                    for: "hidden_states"
                )?.multiArrayValue,

            let presentKV =
                output.featureValue(
                    for: "present_kv"
                )?.multiArrayValue
        else {

            throw NSError(
                domain: "LLMManager",
                code: -3,
                userInfo: [
                    NSLocalizedDescriptionKey:
                        "Output parse failed"
                ]
            )
        }

        return (
            hidden_states: hiddenStates,
            presentKV: presentKV
        )
    }
}
