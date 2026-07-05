import SwiftUI

struct ContentView: View {
    @State private var appState = AppState()

    var body: some View {
        Group {
            if let error = appState.loadError {
                ErrorView(message: error, retry: { appState.loadBrowserData() })
            } else if appState.browserData == nil {
                ProgressView("Loading catalog…")
                    .onAppear { appState.loadBrowserData() }
            } else {
                NavigationSplitView {
                    SidebarView()
                } detail: {
                    if appState.selectedSection.isHome {
                        HomeView()
                    } else {
                        BrowseView()
                    }
                }
                .navigationSplitViewStyle(.balanced)
            }
        }
        .environment(appState)
        .frame(minWidth: 900, minHeight: 600)
    }
}

struct ErrorView: View {
    let message: String
    let retry: () -> Void

    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange)
            Text("Could not load model catalog")
                .font(.title2)
                .fontWeight(.semibold)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
            Button("Retry", action: retry)
                .buttonStyle(.borderedProminent)
        }
        .padding(40)
    }
}

#Preview {
    ContentView()
}
