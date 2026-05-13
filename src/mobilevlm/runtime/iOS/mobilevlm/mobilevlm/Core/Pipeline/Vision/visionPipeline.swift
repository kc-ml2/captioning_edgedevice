import SwiftUI
import CoreML


func encodeImage(
    _ uiImage: UIImage
) -> MLMultiArray? {

    // =========================
    // preprocessing
    // =========================
    
    guard let tensor = preprocessImage(
        uiImage
    ) else {

        print("❌ Preprocessing failed")
        return nil
    }

    // =========================
    // vision encoder
    // =========================
    
    guard let visionOut = runVisionTower(
        mlInput: tensor
    ) else {

        print("❌ Vision encoder failed")
        return nil
    }

    // =========================
    // projector
    // =========================

    guard let projectorOut = runProjector(
        mlInput: visionOut
    ) else {

        print("❌ Projector failed")
        return nil
    }

    return projectorOut
}
