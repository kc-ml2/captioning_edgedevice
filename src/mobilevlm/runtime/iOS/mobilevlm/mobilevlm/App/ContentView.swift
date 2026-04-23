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
                DispatchQueue.global(qos: .userInitiated).async {
                    processImage()
                }
            }
    }

    func processImage() {
        
        guard let uiImage = UIImage(named: "000000000139"),
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

        guard let squareImage = rgbToUIImage(
            rgb: squareRGB,
            width: squareSize,
            height: squareSize
        ) else {
            print("❌ RGB → UIImage 실패")
            return
        }

        guard let resized = resizeWithCI(squareImage) else {
            print("❌ Resize 실패")
            return
        }

        guard let resizedRGB = extractRGB(from: resized) else {
            print("❌ RGB extraction 실패")
            return
        }
        
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
        print("input strides:", mlInput.strides)
        
        guard let features = runVisionTower(mlInput: mlInput) else {
            return
        }

        saveMLMultiArray(features, filename: "vision_out_coreml.bin")
        
        print("✅ Done")
    }
}
