import CoreML

func runProjector(
    mlInput: MLMultiArray
) -> MLMultiArray? {

    do {
        // =========================
        // 1. Model Load
        // =========================
        let config = MLModelConfiguration()
        config.computeUnits = .all

        let model = try Projector_fp32(configuration: config)

        // =========================
        // 2. Inference
        // =========================
        let input = Projector_fp32Input(image_features: mlInput)
        let output = try model.prediction(input: input)

        let projected = output.projected_features


        return projected

    } catch {
        print("❌ Projector inference failed:", error)
        return nil
    }
}
