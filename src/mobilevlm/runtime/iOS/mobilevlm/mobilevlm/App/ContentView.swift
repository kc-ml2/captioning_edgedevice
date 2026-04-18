import SwiftUI
import PhotosUI

struct ContentView: View {

    @State private var selectedImage: UIImage?
    // pickerItem: image saving variable
    @State private var pickerItem: PhotosPickerItem?
    @State private var hasRun = false

    var body: some View {
        Text("Processing...")
            .onAppear {
                processImage()
                    
            }
    }

    func processImage() {
        
        let start = CFAbsoluteTimeGetCurrent()

        
        guard let uiImage = UIImage(named: "000000000139") else {
            print("❌ Image load failed")
            return
        }

        guard let inputTensor = preprocessImage(uiImage) else {
            print("❌ Preprocess failed")
            return
        }
        
        let end = CFAbsoluteTimeGetCurrent()
        print("⏱ Total time: \((end - start) * 1000) ms")

            
        saveToDocuments(inputTensor, filename: "swift_chw.bin")

        print("✅ Done")
    }
}
