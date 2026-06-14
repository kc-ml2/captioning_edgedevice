//import CoreML
//
//
//func saveUInt8ToDocuments(_ array: [UInt8], filename: String) {
//
//    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent(filename)
//
//    let data = array.withUnsafeBufferPointer {
//        Data(buffer: $0)
//    }
//
//    do {
//        try data.write(to: url)
//        print("✅ Saved to:", url)
//    } catch {
//        print("❌ Save failed:", error)
//    }
//}
//
//func saveFP32ToDocuments(_ array: [Float32], filename: String) {
//    
//    // Documents
//    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent(filename)
//    
//    // Float32 → Data (binary)
//    let data = array.withUnsafeBufferPointer {
//        Data(buffer: $0)
//    }
//    
//    do {
//        try data.write(to: url)
//        print("✅ Saved to:", url)
//    } catch {
//        print("❌ Save failed:", error)
//    }
//}
//
//func saveMLMultiArray(_ array: MLMultiArray, filename: String) {
//
//    let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
//        .appendingPathComponent(filename)
//
//    let count = array.count
//
//    if array.dataType == .float32 {
//        let ptr = array.dataPointer.bindMemory(to: Float32.self, capacity: count)
//
//        let data = Data(bytes: ptr, count: count * MemoryLayout<Float32>.size)
//
//        do {
//            try data.write(to: url)
//            print("✅ Saved (fp32) to:", url)
//        } catch {
//            print("❌ Save failed:", error)
//        }
//
//    } else {
//        print("❌ dtype mismatch: expected float32, got \(array.dataType)")
//    }
//}
//
