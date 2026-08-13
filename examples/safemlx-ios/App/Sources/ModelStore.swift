import Foundation
import HuggingFace

struct CachedModel: Codable, Identifiable, Hashable {
    let repoID: String
    let snapshotPath: String

    var id: String { repoID }
    var snapshotURL: URL { URL(fileURLWithPath: snapshotPath, isDirectory: true) }
}

@MainActor
final class ModelStore: ObservableObject {
    @Published var repository = "mlx-community/LFM2.5-1.2B-Instruct-4bit"
    @Published private(set) var models: [CachedModel] = []
    @Published var selectedModelID: String?
    @Published var prompt = "Briefly explain why unified memory is useful for machine learning."
    @Published private(set) var output = ""
    @Published private(set) var status = "Ready"
    @Published private(set) var downloadProgress: Double?
    @Published private(set) var isDownloading = false
    @Published private(set) var isGenerating = false

    private let cache: HubCache
    private let hub: HubClient
    private var engine: SafeMLXEngine?
    private var loadedSnapshotPath: String?
    private static let recordsKey = "cachedModels.v1"

    init() {
        let cacheRoot = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("huggingface", isDirectory: true)
            .appendingPathComponent("hub", isDirectory: true)
        let cache = HubCache(cacheDirectory: cacheRoot)
        self.cache = cache
        self.hub = HubClient(cache: cache)
        restoreModels()
    }

    var selectedModel: CachedModel? {
        models.first { $0.id == selectedModelID }
    }

    func download() async {
        guard !isDownloading, !isGenerating else { return }
        let requested = repository.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let repoID = Repo.ID(rawValue: requested) else {
            status = "Enter a repository as owner/name"
            return
        }

        isDownloading = true
        downloadProgress = 0
        status = "Reading repository…"
        defer {
            isDownloading = false
            downloadProgress = nil
        }

        do {
            let snapshot = try await hub.downloadSnapshot(
                of: repoID,
                matching: ["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
                maxConcurrentDownloads: 3,
                progressHandler: { [weak self] progress in
                    self?.downloadProgress = progress.fractionCompleted
                    self?.status = "Downloading \(Int(progress.fractionCompleted * 100))%"
                }
            )
            guard FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("config.json").path
            ) else {
                throw CocoaError(.fileNoSuchFile, userInfo: [
                    NSLocalizedDescriptionKey: "The snapshot has no root config.json"
                ])
            }
            let model = CachedModel(repoID: requested, snapshotPath: snapshot.path)
            models.removeAll { $0.repoID == requested }
            models.append(model)
            models.sort { $0.repoID.localizedCaseInsensitiveCompare($1.repoID) == .orderedAscending }
            selectedModelID = model.id
            saveModels()
            status = "Cached \(requested)"
        } catch {
            status = "Download failed: \(error.localizedDescription)"
        }
    }

    func delete(_ model: CachedModel) {
        guard !isDownloading, !isGenerating else { return }
        if loadedSnapshotPath == model.snapshotPath {
            engine = nil
            loadedSnapshotPath = nil
        }
        if let repoID = Repo.ID(rawValue: model.repoID) {
            try? FileManager.default.removeItem(at: cache.repoDirectory(repo: repoID, kind: .model))
        }
        models.removeAll { $0.id == model.id }
        if selectedModelID == model.id {
            selectedModelID = models.first?.id
        }
        saveModels()
        status = "Removed \(model.repoID)"
    }

    func generate() async {
        guard !isGenerating, !isDownloading else { return }
        guard let selectedModel else {
            status = "Download and select a model first"
            return
        }
        let submittedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !submittedPrompt.isEmpty else {
            status = "Enter a prompt"
            return
        }

        isGenerating = true
        output = ""
        defer { isGenerating = false }
        do {
            if engine == nil || loadedSnapshotPath != selectedModel.snapshotPath {
                status = "Loading \(selectedModel.repoID)…"
                engine = try await SafeMLXEngine.load(modelAt: selectedModel.snapshotURL)
                loadedSnapshotPath = selectedModel.snapshotPath
            }
            status = "Generating…"
            guard let engine else { return }
            try await engine.generate(prompt: submittedPrompt) { [weak self] fragment in
                Task { @MainActor in
                    self?.output.append(fragment)
                }
            }
            status = "Finished"
        } catch {
            status = "Generation failed: \(error.localizedDescription)"
            engine = nil
            loadedSnapshotPath = nil
        }
    }

    private func restoreModels() {
        guard let data = UserDefaults.standard.data(forKey: Self.recordsKey),
              let decoded = try? JSONDecoder().decode([CachedModel].self, from: data)
        else { return }
        models = decoded.filter {
            FileManager.default.fileExists(
                atPath: $0.snapshotURL.appendingPathComponent("config.json").path
            )
        }
        selectedModelID = models.first?.id
        saveModels()
    }

    private func saveModels() {
        guard let data = try? JSONEncoder().encode(models) else { return }
        UserDefaults.standard.set(data, forKey: Self.recordsKey)
    }
}
