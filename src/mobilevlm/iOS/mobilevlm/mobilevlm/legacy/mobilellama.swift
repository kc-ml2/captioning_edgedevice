//import CoreML
//
//func runmobilellama(
//    inputsEmbeds: MLMultiArray,
//    attentionMask: MLMultiArray,
//    kvCache: MLMultiArray
//) -> (
//    logits: MLMultiArray,
//    presentKV: MLMultiArray
//)? {
//
//    do {
//
//        let model =
//            ModelManager.shared.mobilellamaModel
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
//            try model.model.prediction(
//                from: provider
//            )
//
//        guard
//            let logits =
//                prediction.featureValue(
//                    for: "logits"
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
//            logits: logits,
//            presentKV: presentKV
//        )
//
//    } catch {
//
//        print(error)
//        return nil
//    }
//}
