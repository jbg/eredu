import SwiftUI

@main
struct SafeMLXDemoApp: App {
    @StateObject private var modelStore = ModelStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
        }
    }
}
