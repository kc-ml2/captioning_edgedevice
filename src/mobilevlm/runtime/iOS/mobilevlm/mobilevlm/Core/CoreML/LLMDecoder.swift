import CoreML

func runLLM(
    model: mobilevlm_dynamic,
    inputsEmbeds: MLMultiArray,
    attentionMask: MLMultiArray,
    pastKeyValues: [KVCache]
) -> (
    logits: MLMultiArray,
    presentKeyValues: [KVCache]
)? {

    do {

        // =========================
        // Input Dictionary
        // =========================

        var inputDict: [String: Any] = [:]

        // inputs_embeds
        inputDict["inputs_embeds"] = inputsEmbeds

        // attention_mask
        inputDict["attention_mask"] = attentionMask

        // KV cache
        for layer in 0..<24 {

            inputDict["past_key_\(layer)"] =
                pastKeyValues[layer].key

            inputDict["past_value_\(layer)"] =
                pastKeyValues[layer].value
        }

        // =========================
        // Feature Provider
        // =========================

        let provider = try MLDictionaryFeatureProvider(
            dictionary: inputDict
        )

        // =========================
        // Prediction
        // =========================

        let prediction = try model.model.prediction(
            from: provider
        )

        // =========================
        // Logits
        // =========================

        guard let logits =
            prediction.featureValue(
                for: "logits"
            )?.multiArrayValue
        else {

            print("❌ Failed to get logits")
            return nil
        }

        // =========================
        // Present KV Cache
        // =========================

        var presentKV: [KVCache] = []

        for layer in 0..<24 {

            guard
                let key =
                    prediction.featureValue(
                        for: "present_key_\(layer)"
                    )?.multiArrayValue,

                let value =
                    prediction.featureValue(
                        for: "present_value_\(layer)"
                    )?.multiArrayValue
            else {

                print("❌ Missing KV output at layer \(layer)")
                return nil
            }

            presentKV.append(
                KVCache(
                    key: key,
                    value: value,
                    validLength: 0
                )
            )
        }

        // =========================
        // Return
        // =========================

        return (
            logits: logits,
            presentKeyValues: presentKV
        )

    } catch {

        print("❌ LLM inference failed")
        print(error)

        return nil
    }
}
