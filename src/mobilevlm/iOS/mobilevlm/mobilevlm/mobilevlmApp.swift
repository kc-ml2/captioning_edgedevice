import SwiftUI

@main
struct mobilevlmApp: App {

    init() {

        // =========================
        // CoreML Model Preload
        // =========================

        _ = ModelManager.shared

        // =========================
        // Buffer Preload
        // =========================

        _ = AppBufferManager.shared
    }

    var body: some Scene {

        WindowGroup {

            ContentView()
        }
    }
}
