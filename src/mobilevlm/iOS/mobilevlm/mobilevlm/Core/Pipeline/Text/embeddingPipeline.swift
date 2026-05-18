import CoreML

func buildInputEmbeddings(
    question: String,
    imageFeatures: MLMultiArray
) -> MLMultiArray? {

    // =========================
    // tokenize
    // =========================

    guard let inputIds = tokenizeImageQuestion(
        question: question
    ) else {

        print("❌ Tokenization failed")
        return nil
    }

    // =========================
    // shared embedding weights
    // =========================

    let embeddingWeights: [Int8] =
        ModelManager.shared.embeddingWeights

    // =========================
    // multimodal embeddings
    // =========================

    guard let embeddings =
        buildMultimodalEmbeddings(
            inputIds: inputIds,
            imageFeatures: imageFeatures,
            embeddingWeights: embeddingWeights
        ) else {

        print("❌ Failed to build embeddings")
        return nil
    }

    return embeddings
}
