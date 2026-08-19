import SwiftUI

@main
struct EreduDemoApp: App {
    @StateObject private var modelStore = ModelStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
        }
    }
}
