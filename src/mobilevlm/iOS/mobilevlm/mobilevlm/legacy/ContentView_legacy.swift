//// ContentView.swift
//
//import SwiftUI
//import PhotosUI
//import CoreML
//
//
//struct ContentView: View {
//
//    @State private var pickerItem: PhotosPickerItem?
//
//    @State private var selectedImage: UIImage?
//
//    @State private var questionText: String =
//        "What objects are visible in the image."
//
//    @State private var captionText: String =
//        "Processing..."
//
//    @State private var hasRun = false
//
//    var body: some View {
//
//        VStack(spacing: 20) {
//
//            // =========================
//            // Image
//            // =========================
//
//            if let image = selectedImage {
//
//                Image(uiImage: image)
//                    .resizable()
//                    .scaledToFit()
//                    .frame(height: 300)
//                    .cornerRadius(12)
//            }
//
//            // =========================
//            // Question
//            // =========================
//
//            VStack(alignment: .leading, spacing: 8) {
//
//                Text("Question")
//                    .font(.headline)
//
//                Text(questionText)
//                    .frame(maxWidth: .infinity,
//                           alignment: .leading)
//            }
//            .padding()
//            .background(Color.gray.opacity(0.1))
//            .cornerRadius(12)
//
//            // =========================
//            // Caption
//            // =========================
//
//            VStack(alignment: .leading, spacing: 8) {
//
//                Text("Caption")
//                    .font(.headline)
//
//                Text(captionText)
//                    .frame(maxWidth: .infinity,
//                           alignment: .leading)
//            }
//            .padding()
//            .background(Color.blue.opacity(0.1))
//            .cornerRadius(12)
//
//            Spacer()
//        }
//        .padding()
//        .onAppear {
//
//            if !hasRun {
//
//                hasRun = true
//
//                DispatchQueue.global(
//                    qos: .userInitiated
//                ).async {
//
//                    processmultimodal()
//                }
//            }
//        }
//    }
//
//    func processmultimodal() {
//
//        guard let uiImage = UIImage(named: "image") else {
//
//            print("❌ Image load failed")
//            return
//        }
//
//        // =========================
//        // UI image update
//        // =========================
//
//        DispatchQueue.main.async {
//
//            self.selectedImage = uiImage
//        }
//
//        
////        let width = cgImage.width
////        let height = cgImage.height
////        let squareSize = max(width, height)
//       
//
////        guard let rgb = extractRGB(from: uiImage) else {
////            print("❌ Preprocess failed")
////            return
////        }
////        
////       
////        let squareRGB = expandToSquare(
////            rgb: rgb,
////            width: width,
////            height: height,
////        )
////        
////        let resizedRGB = resizeBilinear336(
////            img: squareRGB,
////            size: squareSize,
////        )
////        
////
////        
////        let tensor = normalizeAndConvertToCHW(
////            rgb: resizedRGB,
////            width: 336,
////            height: 336
////        )
////
////        
////
////        guard let mlInput = makeMLMultiArraySafe(
////            from: tensor,
////            height: 336,
////            width: 336
////        ) else {
////            return
////        }
//        
//        guard let tensor = preprocessImage(uiImage) else {
//            print("❌ Preprocessing failed")
//            return
//        }
//
//        guard let visionOut = runVisionTower(mlInput: tensor) else {
//            return
//        }
//
//        guard let projectorOut = runProjector(mlInput: visionOut) else {
//            print("❌ Projector failed")
//            return
//        }
//        
//        let question = "What objects are visible in the image."
//        
////        let prompt = buildPrompt(question: question)
////        
////        guard let ids = tokenizeImagePrompt(prompt: prompt) else {
////            print("❌ Tokenization failed")
////            return
////        }
////        
//        guard let inputIds = tokenizeImageQuestion(
//            question: question
//        ) else {
//
//            print("❌ Tokenization failed")
//            return
//        }
//        
//        let embeddingWeights = loadEmbeddingWeights()
//        
//        guard let multimodalEmbeddings = buildMultimodalEmbeddings(
//            inputIds: inputIds,
//            imageFeatures: projectorOut,
//            embeddingWeights: embeddingWeights
//        ) else {
//            print("❌ Failed to build multimodal embeddings")
//            return
//        }
//        
//        var generatedTokens: [Int] = []
//        
//        // Shape already: [1, 194, 2048]
//        let curEmbed = multimodalEmbeddings
//
//        // Current sequence length
//        var curPos = curEmbed.shape[1].intValue  // 194
//
//        
//        // Dummy KV cache
//        let kvCache = createEmptyKVCache(batchSize: 1)
//
//        
//        guard let attentionMask = makePrefillAttentionMask(
//            curPos: curPos + 1
//        ) else {
//            return
//        }
//        
//        var llmModel: mobilevlm_dynamic?
//
//        do {
//
//            let config = MLModelConfiguration()
//            config.computeUnits = .all
//
//            llmModel = try mobilevlm_dynamic(
//                configuration: config
//            )
//
//            print("✅ Model loaded")
//
//        } catch {
//
//            print(error)
//            return
//        }
//
//        guard let model = llmModel else {
//            return
//        }
//
//        guard let result = runLLM(
//            model: model,
//            inputsEmbeds: curEmbed,
//            attentionMask: attentionMask,
//            pastKeyValues: kvCache
//        ) else {
//            return
//        }
//        
//        let logits = result.logits
//        
//
//        let nextToken = getNextToken(
//            logits: logits
//        )
//
////        print("✅ next token:", nextToken)
//        
//        var currentKV = result.presentKeyValues
//        
//        guard var nextEmbed = buildNextTokenEmbedding(
//            tokenId: nextToken,
//            embeddingWeights: embeddingWeights
//        ) else {
//            return
//        }
//        
//        generatedTokens.append(nextToken)
//
//        
//        let maxNewTokens = 40
//        let eosTokenId = 2
//
//        for _ in 0..<(maxNewTokens - 1) {
//
//            guard let attentionMask = makePrefillAttentionMask(
//                curPos: curPos + 2
//            ) else {
//                return
//            }
//
//            // dummy KV masking
//            let maskPtr = attentionMask.dataPointer.bindMemory(
//                to: Int32.self,
//                capacity: attentionMask.count
//            )
//            maskPtr[0] = 0
//            
//            guard let result = runLLM(
//                model: model,
//                inputsEmbeds: nextEmbed,
//                attentionMask: attentionMask,
//                pastKeyValues: currentKV
//            ) else {
//                return
//            }
//            
//            let logits = result.logits
//            
//
//            let nextToken = getNextToken(
//                logits: logits
//            )
//
////            print("✅ next token:", nextToken)
//            
//            currentKV = result.presentKeyValues
//            
//            guard let newEmbed = buildNextTokenEmbedding(
//                tokenId: nextToken,
//                embeddingWeights: embeddingWeights
//            ) else {
//                return
//            }
//
//            nextEmbed = newEmbed
//            let nextPos = curPos + 1
//            
//            generatedTokens.append(nextToken)
//            
//            if nextToken == eosTokenId {
//                break
//            }
//            
//            curPos = nextPos
//            
//        }
//        
//        guard let caption = decodeTokens(
//            generatedTokens: generatedTokens
//        ) else {
//            return
//        }
//
//        print("caption:", caption)
//        
//        // =========================
//        // UI update
//        // =========================
//
//        DispatchQueue.main.async {
//
//            self.questionText = question
//            self.captionText = caption
//        }
//        
//        
//        print("Done")
//    }
//}
