import SwiftUI
import CoreML


func encodeImage(
    _ uiImage: UIImage
) -> MLMultiArray {
    
    // =========================
    // preprocessing
    // =========================
    
    let preprocessStart =
    CFAbsoluteTimeGetCurrent()
    
    let tensor = preprocessImage(
        uiImage
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
    
    guard let visionOut = runVisionEnc(
        mlInput: tensor
    ) else {
        
        fatalError("❌ Vision encoder failed")
    }
    
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

        fatalError("❌ Projector failed")
    }

    print(String(
        format: "⏱ Projector Time: %.3f sec",
        CFAbsoluteTimeGetCurrent() - projectorStart
    ))

    return projectorOut
}
