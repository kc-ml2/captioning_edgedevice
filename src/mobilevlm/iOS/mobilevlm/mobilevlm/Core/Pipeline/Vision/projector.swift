import CoreML

func runProjector(
    mlInput: MLMultiArray
) -> MLMultiArray? {

    do {

        let model =
            ModelManager.shared.projectorModel

        let input = Projector_int8Input(
            image_features: mlInput
        )

        let output = try model.prediction(
            input: input
        )

        return output.projected_features

    } catch {

        print(error)
        return nil
    }
}
