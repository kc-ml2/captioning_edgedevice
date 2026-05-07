// ContentView.swift

import SwiftUI
import PhotosUI
import CoreML


struct ContentView: View {

    
    @State private var pickerItem: PhotosPickerItem?
    @State private var hasRun = false

    var body: some View {
        Text("Processing...")
            .onAppear {
                DispatchQueue.global(qos: .userInitiated).async {
                    processmultimodal()
                }
            }
    }

    func processmultimodal() {
        
        
        
        guard let uiImage = UIImage(named: "image"),
              let cgImage = uiImage.cgImage else {
            print("❌ Image load failed")
            return
        }

        let width = cgImage.width
        let height = cgImage.height
        let squareSize = max(width, height)
       

        guard let rgb = extractRGB(from: uiImage) else {
            print("❌ Preprocess failed")
            return
        }
        
       
        let squareRGB = expandToSquare(
            rgb: rgb,
            width: width,
            height: height,
        )
        
        let resizedRGB = resizeBilinear336(
            img: squareRGB,
            size: squareSize,
        )
        

        
        let tensor = normalizeAndConvertToCHW(
            rgb: resizedRGB,
            width: 336,
            height: 336
        )

        

        guard let mlInput = makeMLMultiArraySafe(
            from: tensor,
            height: 336,
            width: 336
        ) else {
            return
        }

        guard let visionOut = runVisionTower(mlInput: mlInput) else {
            return
        }

        guard let projectorOut = runProjector(mlInput: visionOut) else {
            print("❌ Projector failed")
            return
        }
        
        let question = "What objects are visible in the image."
        let prompt = buildPrompt(question: question)
        let embeddingWeights = loadEmbeddingWeights()
        
        guard let ids = tokenizeImagePrompt(prompt: prompt) else {
            print("❌ Tokenization failed")
            return
        }
        
        guard let multimodalEmbeddings = buildMultimodalEmbeddings(
            inputIds: ids,
            imageFeatures: projectorOut,
            embeddingWeights: embeddingWeights
        ) else {
            print("❌ Failed to build multimodal embeddings")
            return
        }
        
        // Shape already: [1, 194, 2048]
        let curEmbed = multimodalEmbeddings

        // Current sequence length
        let curLen = curEmbed.shape[1].intValue  // 194
        
        // Empty KV cache
        let kvCache = createEmptyKVCache(batchSize: 1)

        // Flatten KV tensors
        var pastKeyValues: [MLMultiArray] = []

        for cache in kvCache {

            pastKeyValues.append(cache.key)
            pastKeyValues.append(cache.value)
        }

        // Generated tokens
//        var generatedTokens: [Int] = []
        
        guard let attentionMask = buildAttentionMask(curLen: curLen) else {
            return
        }
        
        guard let llmOutput = runLLMDecoder(
            inputsEmbeds: curEmbed,
            attentionMask: attentionMask,
            pastKeyValues: pastKeyValues
        ) else {

            return
        }

        let nextToken = extractNextToken(
            logits: llmOutput.logits
        )

        print("next token =", nextToken)



        
//        saveUInt8ToDocuments(resizedRGB, filename: "swift_manual_resize.bin")
//        saveFP32ToDocuments(tensor, filename: "swift_tensor.bin")
//        saveMLMultiArray(projectorOut, filename: "projector_coreml.bin")
        
        print("Done")
    }
}
