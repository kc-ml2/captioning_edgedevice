import SwiftUI
import CoreML


func encodeImage(
    _ uiImage: UIImage
) -> MLMultiArray? {

    // =========================
    // preprocessing
    // =========================
    
    let preprocessStart =
        CFAbsoluteTimeGetCurrent()

    guard let tensor = preprocessImage(
        uiImage
    ) else {

        print("❌ Preprocessing failed")
        return nil
    }
    
    inspectArray(
        name: "tensor",
        array: tensor
    )
    
    print(String(
        format: "⏱ Preprocess Time: %.3f sec",
        CFAbsoluteTimeGetCurrent() - preprocessStart
    ))

    // =========================
    // vision encoder
    // =========================
    
    let visionStart =
        CFAbsoluteTimeGetCurrent()
    
    guard let visionOut = runVisionTower(
        mlInput: tensor
    ) else {

        print("❌ Vision encoder failed")
        return nil
    }
    
    inspectArray(
        name: "visionOut",
        array: visionOut
    )
    
     print(String(
         format: "⏱ Vision Encoder Time: %.3f sec",
         CFAbsoluteTimeGetCurrent() - visionStart
     ))

    // =========================
    // projector
    // =========================
    
    let projectorStart =
        CFAbsoluteTimeGetCurrent()

    guard let projectorOut = runProjector(
        mlInput: visionOut
    ) else {

        print("❌ Projector failed")
        return nil
    }

    print(String(
        format: "⏱ Projector Time: %.3f sec",
        CFAbsoluteTimeGetCurrent() - projectorStart
    ))

    return projectorOut
}
