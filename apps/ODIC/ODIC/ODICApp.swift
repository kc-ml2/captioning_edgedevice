//
//  ODICApp.swift
//  ODIC
//
//  Created by softmax on 8/21/26.
//

import SwiftUI

@main
struct ODICApp: App {
    @Environment(\.scenePhase) private var scenePhase
    var body: some Scene {
        WindowGroup {
            ContentView()
                .task { await BenchmarkTelemetry.shared.start() }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .active {
                        Task { await BenchmarkTelemetry.shared.start() }
                    }
                }
        }
    }
}
