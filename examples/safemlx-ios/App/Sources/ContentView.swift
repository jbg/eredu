import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: ModelStore

    var body: some View {
        NavigationStack {
            Form {
                Section("Hugging Face model") {
                    TextField("owner/model", text: $store.repository)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                    Button("Download to cache") {
                        Task { await store.download() }
                    }
                    .disabled(store.isDownloading || store.isGenerating)
                    if let progress = store.downloadProgress {
                        ProgressView(value: progress)
                    }
                }

                Section("Cached models") {
                    if store.models.isEmpty {
                        Text("No models cached")
                            .foregroundStyle(.secondary)
                    } else {
                        Picker("Model", selection: $store.selectedModelID) {
                            ForEach(store.models) { model in
                                Text(model.repoID).tag(Optional(model.id))
                            }
                        }
                        ForEach(store.models) { model in
                            HStack {
                                Text(model.repoID)
                                    .font(.caption)
                                    .lineLimit(1)
                                Spacer()
                                Button(role: .destructive) {
                                    store.delete(model)
                                } label: {
                                    Image(systemName: "trash")
                                }
                                .buttonStyle(.borderless)
                                .disabled(store.isDownloading || store.isGenerating)
                            }
                        }
                    }
                }

                Section("Prompt") {
                    TextEditor(text: $store.prompt)
                        .frame(minHeight: 100)
                    Button(store.isGenerating ? "Generating…" : "Generate") {
                        Task { await store.generate() }
                    }
                    .disabled(store.selectedModel == nil || store.isDownloading || store.isGenerating)
                }

                Section("Output") {
                    ScrollView {
                        Text(store.output.isEmpty ? "Output streams here." : store.output)
                            .foregroundStyle(store.output.isEmpty ? .secondary : .primary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .textSelection(.enabled)
                    }
                    .frame(minHeight: 180)
                }

                Section {
                    Text(store.status)
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("SafeMLX")
        }
    }
}
