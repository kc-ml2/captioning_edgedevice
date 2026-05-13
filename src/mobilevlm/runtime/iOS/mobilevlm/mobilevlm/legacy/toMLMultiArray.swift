// import CoreML

// func makeMLMultiArraySafe(
//     from chw: [Float],
//     height: Int,
//     width: Int
// ) -> MLMultiArray? {

//     let expected = 3 * height * width
//     guard chw.count == expected else {
//         print("❌ Invalid tensor size")
//         return nil
//     }

//     do {
//         let mlArray = try MLMultiArray(
//             shape: [1, 3, height as NSNumber, width as NSNumber],
//             dataType: .float32
//         )

//         // 🔥 stride-safe 방식으로 채우기
//         let H = height
//         let W = width

//         for c in 0..<3 {
//             for h in 0..<H {
//                 for w in 0..<W {

//                     let chwIndex = c * H * W + h * W + w

//                     let index: [NSNumber] = [
//                         0,
//                         NSNumber(value: c),
//                         NSNumber(value: h),
//                         NSNumber(value: w)
//                     ]

//                     mlArray[index] = NSNumber(value: chw[chwIndex])
//                 }
//             }
//         }

//         return mlArray

//     } catch {
//         print("❌ MLMultiArray 생성 실패:", error)
//         return nil
//     }
// }

// func mlMultiArrayToFloatArray(_ mlArray: MLMultiArray) -> [Float] {
//     let count = mlArray.count
//     var result = [Float](repeating: 0, count: count)

//     for i in 0..<count {
//         result[i] = mlArray[i].floatValue
//     }

//     return result
// }
