//
//  LLMDecoder.swift
//

import Foundation
import CoreML

// =====================================================
// MARK: - Constants
// =====================================================

private let NUM_LAYERS = 24

// =====================================================
// MARK: - Output Struct
// =====================================================

struct LLMOutput {

    let logits: MLMultiArray
    let presentKeyValues: [MLMultiArray]
}

// =====================================================
// MARK: - LLM Decoder
// =====================================================

func runLLMDecoder(
    inputsEmbeds: MLMultiArray,
    attentionMask: MLMultiArray,
    pastKeyValues: [MLMultiArray]
) -> LLMOutput? {

    do {

        // =================================================
        // 1. Load Model
        // =================================================

        let config = MLModelConfiguration()

        config.computeUnits = .all

        let model = try MobileLlamaDecoder(
            configuration: config
        )

        // =================================================
        // 2. Build Input Dictionary
        // =================================================

        var inputs: [String: Any] = [:]

        inputs["inputs_embeds"] =
            inputsEmbeds

        inputs["attention_mask"] =
            attentionMask

        for i in 0..<pastKeyValues.count {

            inputs["past_key_values_\(i)"] =
                pastKeyValues[i]
        }

        // =================================================
        // 3. Feature Provider
        // =================================================

        let provider =
            try MLDictionaryFeatureProvider(
                dictionary: inputs
            )

        // =================================================
        // 4. Prediction
        // =================================================

        let prediction =
            try model.model.prediction(
                from: provider
            )

        // =================================================
        // 5. Get Logits
        // =================================================

        guard let logits =
            prediction.featureValue(
                for: "logits"
            )?.multiArrayValue
        else {

            print("❌ Failed to get logits")
            return nil
        }

        print("✅ logits shape:", logits.shape)

        // =================================================
        // 6. Get Present KV Cache
        // =================================================

        var newKV: [MLMultiArray] = []

        for layer in 0..<NUM_LAYERS {

            // ---------------------------------------------
            // Key
            // ---------------------------------------------

            let keyName =
                "present_key_\(layer)"

            guard let key =
                prediction.featureValue(
                    for: keyName
                )?.multiArrayValue
            else {

                print("❌ Missing \(keyName)")
                return nil
            }

            // ---------------------------------------------
            // Value
            // ---------------------------------------------

            let valueName =
                "present_value_\(layer)"

            guard let value =
                prediction.featureValue(
                    for: valueName
                )?.multiArrayValue
            else {

                print("❌ Missing \(valueName)")
                return nil
            }

            newKV.append(key)
            newKV.append(value)
        }

        print("✅ KV cache count:", newKV.count)

        // =================================================
        // 7. Return
        // =================================================

        return LLMOutput(
            logits: logits,
            presentKeyValues: newKV
        )

    } catch {

        print("❌ LLM inference failed:", error)
        return nil
    }
}

// =====================================================
// MARK: - Greedy Argmax
// =====================================================

func extractNextToken(
    logits: MLMultiArray
) -> Int {

    let seqLen =
        logits.shape[1].intValue

    let vocabSize =
        logits.shape[2].intValue

    let ptr = logits.dataPointer.bindMemory(
        to: Float32.self,
        capacity: logits.count
    )

    let offset =
        (seqLen - 1) * vocabSize

    var maxValue: Float =
        -Float.infinity

    var token: Int = 0

    for i in 0..<vocabSize {

        let value = ptr[offset + i]

        if value > maxValue {

            maxValue = value
            token = i
        }
    }

    return token
}
