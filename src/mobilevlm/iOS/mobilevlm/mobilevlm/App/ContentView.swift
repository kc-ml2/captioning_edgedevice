// ContentView.swift

import SwiftUI
import PhotosUI
import CoreML


struct ContentView: View {

    // =========================
    // UI State
    // =========================
    
    @State private var pickerItem: PhotosPickerItem?

    @State private var selectedImage: UIImage?


    @State private var captionText: String =
        "Select an image."
    
    @State private var isProcessing = false

    var body: some View {

        // =========================
        // UI Layout
        // =========================
        
        VStack(spacing: 20) {
            
            // =========================
            // Image
            // =========================
            
            PhotosPicker(
                selection: $pickerItem,
                matching: .images
            ) {

                Text("Select Image")
                    .padding()
                    .background(Color.blue)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }

            if let image = selectedImage {

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .cornerRadius(12)
            }

            // =========================
            // Caption
            // =========================

            VStack(alignment: .leading, spacing: 8) {

                Text("Caption")
                    .font(.headline)

                Text(captionText)
                    .frame(maxWidth: .infinity,
                           alignment: .leading)
            }
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(12)

            Spacer()
        }
        .padding()
        
        // =========================
        // UI Event Handling
        // =========================
        
        .onChange(of: pickerItem) {

            Task {

                guard let item = pickerItem else {
                    return
                }

                guard let data = try? await item.loadTransferable(
                    type: Data.self
                ) else {
                    return
                }

                guard let uiImage = UIImage(data: data) else {
                    return
                }

                DispatchQueue.main.async {

                    self.selectedImage = uiImage

                    guard !self.isProcessing else {
                        return
                    }

                    self.isProcessing = true
                    self.captionText = "Processing..."

                    Task(priority: .userInitiated) {

                        processmultimodal(
                            uiImage: uiImage
                        )
                    }
                }
            }
        }
    }

    // =========================
    // Multimodal Processing
    // =========================

    func processmultimodal(
        uiImage: UIImage
    ) {
        
        let totalStart = CFAbsoluteTimeGetCurrent()
        
        defer {

            print(String(
                format: "⏱ Total Time: %.3f sec",
                CFAbsoluteTimeGetCurrent() - totalStart
            ))
            print("========================================")
            
            // =========================
            // Processing State Reset
            // =========================

            DispatchQueue.main.async {

                self.isProcessing = false
            }
        }
        

        // =========================
        // Vision Encoder
        // =========================

        guard let imageFeatures = encodeImage(uiImage)
        else {
            print("❌ Vision pipeline failed")
            return
        }
        
        // =========================
        // Multimodal Embedding
        // =========================

        let mmInputStart = CFAbsoluteTimeGetCurrent()

        let question =
            "What objects are visible in the scene?"

        guard let multimodalEmbeddings =
            buildInputEmbeddings(
                question: question,
                imageFeatures: imageFeatures
            ) else {

            return
        }
        
        print(String(
            format: "⏱ Multimodal Input Time: %.3f sec",
            CFAbsoluteTimeGetCurrent() - mmInputStart
        ))

        var generatedTokens: [Int] = []

        let curEmbed = multimodalEmbeddings
        
        var curPos =
            curEmbed.shape[1].intValue + 1

        let bufferManager = AppBufferManager.shared

        bufferManager.resetKVCache()

        let kvCache = bufferManager.kvCaches
        
        let reusableNextEmbed =
            bufferManager.reusableNextEmbed
        
        // =========================
        // reusableNextEmbed dtype
        // =========================

        guard let attentionMask =
            makePrefillAttentionMask(
                curPos: curPos
            ) else {
            return
        }
        
        // =========================
        // Prefill
        // =========================
        
        let prefillStart = CFAbsoluteTimeGetCurrent()

        guard let result = runLLM(
            inputsEmbeds: curEmbed,
            attentionMask: attentionMask,
            pastKeyValues: kvCache
        ) else {
            return
        }

        print(String(
            format: "⏱ Prefill Time: %.3f sec",
            CFAbsoluteTimeGetCurrent() - prefillStart
        ))

        let logits = result.logits
        
        let nextToken = getNextToken(
            logits: logits
        )
        
        var currentKV = result.presentKeyValues

        updateNextTokenEmbedding(
            tokenId: nextToken,
            embedBuffer: reusableNextEmbed
        )

        generatedTokens.append(nextToken)

        curPos += 1

        let maxNewTokens = 40
        let eosTokenId = 2

        // =========================
        // Decoder
        // =========================
        
        let decoderStart = CFAbsoluteTimeGetCurrent()

        for _ in 0..<(maxNewTokens - 1) {

            guard let attentionMask =
                makePrefillAttentionMask(
                    curPos: curPos
                ) else {
                return
            }


            guard let result = runLLM(
                inputsEmbeds: reusableNextEmbed,
                attentionMask: attentionMask,
                pastKeyValues: currentKV
            ) else {
                return
            }

            let logits = result.logits

            let nextToken = getNextToken(
                logits: logits
            )
            
            currentKV = result.presentKeyValues

            updateNextTokenEmbedding(
                tokenId: nextToken,
                embedBuffer: reusableNextEmbed
            )

            generatedTokens.append(nextToken)

            if nextToken == eosTokenId {
                break
            }

            curPos += 1
        }

        print(String(
            format: "⏱ Avg Decoder Time per Toekn: %.3f sec",
            (CFAbsoluteTimeGetCurrent() - decoderStart) / Double(generatedTokens.count)
        ))

        // =========================
        // Decode Tokens
        // =========================

        guard let caption = decodeTokens(
            generatedTokens: generatedTokens
        ) else {
            return
        }
        
        // =========================
        // UI update
        // =========================

        DispatchQueue.main.async {

            self.captionText = caption
        }
    }
}
