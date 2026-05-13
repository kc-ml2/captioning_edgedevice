// ContentView.swift

import SwiftUI
import PhotosUI
import CoreML


struct ContentView: View {

    // =========================
    // UI State
    // =========================

    @State private var picskerItem: PhotosPickerItem?

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
            // Image Picker UI
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


            // =========================
            // Image Preview UI
            // =========================

            if let image = selectedImage {

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .cornerRadius(12)
            }

            // =========================
            // Caption UI
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

        defer {

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

        let question =
            "What objects are visible in the scene?"

        guard let multimodalEmbeddings =
            buildInputEmbeddings(
                question: question,
                imageFeatures: imageFeatures
            ) else {
            return
        }

        // =========================
        // LLM Initialization
        // =========================        

        var generatedTokens: [Int] = []

        let curEmbed = multimodalEmbeddings

        var curPos =
            curEmbed.shape[1].intValue + 1

        let bufferManager = AppBufferManager.shared

        bufferManager.resetKVCache()

        let kvCache = bufferManager.kvCaches
        
        let reusableNextEmbed =
            bufferManager.reusableNextEmbed

        guard let attentionMask =
            makePrefillAttentionMask(
                curPos: curPos
            ) else {
            return
        }
        
        let maskPtr =
            attentionMask.dataPointer.bindMemory(
                to: Int32.self,
                capacity: attentionMask.count
            )

        maskPtr[0] = 0
        

        // =========================
        // Prefill Stage
        // =========================


        guard let result = runLLM(
            inputsEmbeds: curEmbed,
            attentionMask: attentionMask,
            pastKeyValues: kvCache
        ) else {
            return
        }

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
        // Decoder Loop
        // =========================


        for _ in 0..<(maxNewTokens - 1) {

            guard let attentionMask =
                makePrefillAttentionMask(
                    curPos: curPos
                ) else {
                return
            }

            let maskPtr =
                attentionMask.dataPointer.bindMemory(
                    to: Int32.self,
                    capacity: attentionMask.count
                )

            maskPtr[0] = 0

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

        // =========================
        // Token Decoding
        // =========================

        guard let caption = decodeTokens(
            generatedTokens: generatedTokens
        ) else {
            return
        }

        print("caption:", caption)
        
        // =========================
        // UI update
        // =========================

        DispatchQueue.main.async {

            self.captionText = caption
        }
    }
}
