import SwiftUI

@main
struct mobilevlmApp: App {

    init() {
        
        let startTime = CFAbsoluteTimeGetCurrent()
        
        _ = ModelManager.shared
        
        print(
            String(
                format: "⏱ ModelManager Load Time: %.3f sec", CFAbsoluteTimeGetCurrent() - startTime
            )
        )
    }

    var body: some Scene {

        WindowGroup {
            ContentView()
        }
    }
}
