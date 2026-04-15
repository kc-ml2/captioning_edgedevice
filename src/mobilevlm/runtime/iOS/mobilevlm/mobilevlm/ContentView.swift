import SwiftUI
import PhotosUI

struct ContentView: View {
    @State private var selectedImage: UIImage?
    @State private var pickerItem: PhotosPickerItem?
    @State private var userInput: String = ""

    var body: some View {
        VStack(spacing: 20) {

            if let image = selectedImage {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 250)
            }

            PhotosPicker("Select a photo", selection: $pickerItem, matching: .images)

            TextField("Insert text", text: $userInput)
                .textFieldStyle(RoundedBorderTextFieldStyle())
                .padding()

            Text("Question: \(userInput)")
        }
        .padding()
        .onChange(of: pickerItem) { oldValue, newItem in
            Task {
                if let data = try? await newItem?.loadTransferable(type: Data.self),
                   let uiImage = UIImage(data: data) {
                    selectedImage = uiImage
                }
            }
        }
    }
}

#Preview {
    ContentViewPreviewWrapper()
}

struct ContentViewPreviewWrapper: View {
    var body: some View {
        ContentView()
    }
}
