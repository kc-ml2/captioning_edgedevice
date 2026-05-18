import CoreML

func runLLM(
    inputsEmbeds: MLMultiArray,
    attentionMask: MLMultiArray,
    pastKeyValues: [KVCache]
) -> (
    logits: MLMultiArray,
    presentKeyValues: [KVCache]
)? {

    do {

        let model =
            ModelManager.shared.llmModel

        var inputDict: [String: Any] = [:]

        inputDict["inputs_embeds"] =
            inputsEmbeds

        inputDict["attention_mask"] =
            attentionMask

        for layer in 0..<24 {

            inputDict["past_key_\(layer)"] =
                pastKeyValues[layer].key

            inputDict["past_value_\(layer)"] =
                pastKeyValues[layer].value
        }

        let provider =
            try MLDictionaryFeatureProvider(
                dictionary: inputDict
            )

        let prediction =
            try model.model.prediction(
                from: provider
            )

        // =========================
        // logits
        // =========================

        guard let logits =
            prediction.featureValue(
                for: "logits"
            )?.multiArrayValue
        else {

            return nil
        }

        // =========================
        // present kv
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

                return nil
            }

            presentKV.append(
                KVCache(
                    key: key,
                    value: value,
                )
            )
        }

        return (
            logits: logits,
            presentKeyValues: presentKV
        )

    } catch {

        print(error)
        return nil
    }
}
