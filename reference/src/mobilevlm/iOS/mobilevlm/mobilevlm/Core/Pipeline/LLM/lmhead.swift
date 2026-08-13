// lmhead.swift

import Foundation
import CoreML

func lmHead(
    hiddenStates: MLMultiArray
) -> MLMultiArray {

    let result =
        try! ModelManager.shared.lmheadModel.prediction(
            hidden_states: hiddenStates
        )

    return result.logits
}
