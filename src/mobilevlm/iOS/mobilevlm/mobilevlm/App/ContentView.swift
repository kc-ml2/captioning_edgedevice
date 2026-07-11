// ContentView.swift

import SwiftUI
import PhotosUI


struct ContentView: View {
    
    // =========================
    // UI State
    // =========================
    
    @State private var pickerItem: PhotosPickerItem?
    
    @State private var selectedImage: UIImage?
    
    @State private var captionText: String = "Select an image."
    
    @State private var isProcessing = false
    
    @State private var showCamera = false
    
    private let processor = MultimodalProcessor()
    
    var body: some View {
        
        VStack(spacing: 20) {

            if let image = selectedImage {

                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 300)
                    .cornerRadius(12)
            }

            ImagePickerView(
                pickerItem: $pickerItem,
                showCamera: $showCamera
            )
            .padding(.horizontal)

            VStack(alignment: .leading, spacing: 8) {

                Text("Caption")
                    .font(.headline)

                Text(captionText)
                    .frame(
                        maxWidth: .infinity,
                        alignment: .leading
                    )
            }
            .padding()
            .background(Color.blue.opacity(0.1))
            .cornerRadius(12)

            Spacer()
        }
        
        .sheet(isPresented: $showCamera) {

            CameraPicker(
                selectedImage: $selectedImage
            )
        }
        
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
                    
                    selectedImage = uiImage
                }
            }
        }
        
        .onChange(of: selectedImage) {

            guard let image = selectedImage else {
                return
            }

            guard !isProcessing else {
                return
            }

            isProcessing = true

            // First update UI
            captionText = "Processing..."

            // Allow SwiftUI to render selected image
            DispatchQueue.main.asyncAfter(
                deadline: .now() + 0.1
            ) {

                Task(priority: .userInitiated) {

                    let caption =
                        processor.mobilevlm(
                            uiImage: image
                        )

                    await MainActor.run {

                        captionText =
                            caption ?? "Failed"

                        isProcessing = false
                    }
                }
            }
        }
    }
}
