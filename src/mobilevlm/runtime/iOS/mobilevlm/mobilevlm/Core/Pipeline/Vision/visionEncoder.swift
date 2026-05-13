import CoreML

func runVisionTower(
    mlInput: MLMultiArray
) -> MLMultiArray? {

    do {

        let model =
            ModelManager.shared.visionModel

        let input = VisionEncoder_32Input(
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
