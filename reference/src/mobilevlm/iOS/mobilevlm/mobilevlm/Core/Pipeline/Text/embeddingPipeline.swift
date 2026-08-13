// embeddingPipeline.swift


import CoreML

func buildInputEmbeddings(
    question: String = "What objects are visible in the scene?",
    imageFeatures: MLMultiArray
) -> MLMultiArray? {
    
    let mmInputStart = CFAbsoluteTimeGetCurrent()

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
    // multimodal embeddings
    // =========================

    guard let embeddings =
        buildMultimodalEmbeddings(
            inputIds: inputIds,
            imageFeatures: imageFeatures,
        ) else {

        print("❌ Failed to build embeddings")
        return nil
    }
    
    print(String(
        format: "⏱ Multimodal Input Time: %.3f sec",
        CFAbsoluteTimeGetCurrent() - mmInputStart
    ))

    return embeddings
}
