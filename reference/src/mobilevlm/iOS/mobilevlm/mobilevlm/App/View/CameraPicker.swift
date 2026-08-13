// CameraPicker.swift

import SwiftUI
import UIKit


struct CameraPicker: UIViewControllerRepresentable {

    @Binding var selectedImage: UIImage?

    @Environment(\.dismiss) private var dismiss

    func makeCoordinator() -> Coordinator {
        Coordinator(self)
    }

    func makeUIViewController(
        context: Context
    ) -> UIImagePickerController {

        print("CameraPicker created")

        let picker = UIImagePickerController()

        if UIImagePickerController.isSourceTypeAvailable(.camera) {

            print("Camera available")

            picker.sourceType = .camera
            picker.cameraCaptureMode = .photo

        } else {

            print("Camera unavailable")

            picker.sourceType = .photoLibrary
        }

        picker.delegate = context.coordinator

        return picker
    }

    func updateUIViewController(
        _ uiViewController: UIImagePickerController,
        context: Context
    ) {
    }

    class Coordinator:
        NSObject,
        UINavigationControllerDelegate,
        UIImagePickerControllerDelegate {

        let parent: CameraPicker

        init(_ parent: CameraPicker) {
            self.parent = parent
        }

        func imagePickerController(
            _ picker: UIImagePickerController,
            didFinishPickingMediaWithInfo info: [
                UIImagePickerController.InfoKey: Any
            ]
        ) {

            guard let image =
                info[.originalImage] as? UIImage
            else {

                parent.dismiss()
                return
            }

            // Close camera first
            parent.dismiss()

            // Then update image
            DispatchQueue.main.asyncAfter(
                deadline: .now() + 0.3
            ) {

                self.parent.selectedImage = image
            }
        }

        func imagePickerControllerDidCancel(
            _ picker: UIImagePickerController
        ) {

            parent.dismiss()
        }
    }
}
