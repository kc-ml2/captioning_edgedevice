// visionEncoder.swift


import CoreML

func runVisionEnc(
    mlInput: MLMultiArray
) -> MLMultiArray? {

    do {

        let model =
            ModelManager.shared.visionModel

        let input = vit_fp16Input(
            pixel_values: mlInput
        )

        let output = try model.prediction(
            input: input
        )

        return output.image_features

    } catch {

        print(error)
        return nil
    }
}
