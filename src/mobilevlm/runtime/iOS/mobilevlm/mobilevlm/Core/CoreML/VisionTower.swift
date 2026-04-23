import CoreML

func runVisionTower(
    mlInput: MLMultiArray
) -> MLMultiArray? {

    do {
        // =========================
        // 1. Model Load
        // =========================
        let config = MLModelConfiguration()
        config.computeUnits = .all

        let model = try VisionTower_fp32(configuration: config)

        // =========================
        // 2. Inference
        // =========================
        let input = VisionTower_fp32Input(pixel_values: mlInput)
        let output = try model.prediction(input: input)

        let features = output.image_features

        // =========================
        // 3. Debug
        // =========================
        print("✅ Output shape:", features.shape)
        print("✅ Output strides:", features.strides)

        return features

    } catch {
        print("❌ CoreML inference failed:", error)
        return nil
    }
}
