//  ImagePicker.swift


import SwiftUI
import PhotosUI

struct ImagePickerView: View {

    @Binding var pickerItem: PhotosPickerItem?
    @Binding var showCamera: Bool

    var body: some View {

        HStack(spacing: 40) {

            PhotosPicker(
                selection: $pickerItem,
                matching: .images
            ) {

                Image(systemName: "photo.on.rectangle.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.white)
                    .frame(width: 100, height: 100)
                    .background(Color.blue)
                    .clipShape(Circle())
            }

            Button {

                showCamera = true

            } label: {

                Image(systemName: "camera.fill")
                    .font(.system(size: 40))
                    .foregroundColor(.white)
                    .frame(width: 100, height: 100)
                    .background(Color.green)
                    .clipShape(Circle())
            }
        }
    }
}
