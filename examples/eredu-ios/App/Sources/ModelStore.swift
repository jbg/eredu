import Foundation
import HuggingFace

struct CachedModel: Codable, Identifiable, Hashable {
    let repoID: String
    let revision: String

    var id: String { repoID }

    init(repoID: String, revision: String) {
        self.repoID = repoID
        self.revision = revision
    }

    private enum CodingKeys: String, CodingKey {
        case repoID
        case revision
        case snapshotPath
    }

    init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        repoID = try values.decode(String.self, forKey: .repoID)
        if let revision = try values.decodeIfPresent(String.self, forKey: .revision) {
            self.revision = revision
        } else {
            // cachedModels.v1 stored an absolute app-container path. Keep it readable
            // long enough to migrate existing installs to a stable revision identifier.
            let snapshotPath = try values.decode(String.self, forKey: .snapshotPath)
            revision = URL(fileURLWithPath: snapshotPath).lastPathComponent
        }
    }

    func encode(to encoder: Encoder) throws {
        var values = encoder.container(keyedBy: CodingKeys.self)
        try values.encode(repoID, forKey: .repoID)
        try values.encode(revision, forKey: .revision)
    }
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
    private var engine: EreduEngine?
    private var loadedSnapshotPath: String?
    private var loadedModelLoadSeconds: TimeInterval?
    private static let recordsKey = "cachedModels.v1"
    private static let downloadedFileExtensions: Set<String> = [
        "json", "safetensors", "jinja", "model", "txt",
    ]

    init() {
        let fileManager = FileManager.default
        let applicationSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        let huggingFaceDirectory = applicationSupport.appendingPathComponent("huggingface", isDirectory: true)
        let cacheRoot = huggingFaceDirectory.appendingPathComponent("hub", isDirectory: true)

        Self.migrateLegacyCacheIfNeeded(to: huggingFaceDirectory)
        try? fileManager.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
        var resourceValues = URLResourceValues()
        resourceValues.isExcludedFromBackup = true
        var excludedDirectory = huggingFaceDirectory
        try? excludedDirectory.setResourceValues(resourceValues)

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
        downloadProgress = nil
        status = "Reading repository…"
        defer {
            isDownloading = false
            downloadProgress = nil
        }

        do {
            let snapshot = try await downloadSnapshot(repoID)
            guard FileManager.default.fileExists(
                atPath: snapshot.appendingPathComponent("config.json").path
            ) else {
                throw CocoaError(.fileNoSuchFile, userInfo: [
                    NSLocalizedDescriptionKey: "The snapshot has no root config.json"
                ])
            }
            let model = CachedModel(repoID: requested, revision: snapshot.lastPathComponent)
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
        let snapshotPath = snapshotURL(for: model).path
        if loadedSnapshotPath == snapshotPath {
            engine = nil
            loadedSnapshotPath = nil
            loadedModelLoadSeconds = nil
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
            let snapshotURL = snapshotURL(for: selectedModel)
            if engine == nil || loadedSnapshotPath != snapshotURL.path {
                status = "Loading \(selectedModel.repoID)…"
                let loadStartedAt = ProcessInfo.processInfo.systemUptime
                engine = try await EreduEngine.load(modelAt: snapshotURL)
                loadedModelLoadSeconds = ProcessInfo.processInfo.systemUptime - loadStartedAt
                loadedSnapshotPath = snapshotURL.path
            }
            status = "Generating…"
            guard let engine else { return }
            let stats = try await engine.generate(prompt: submittedPrompt) { [weak self] fragment in
                Task { @MainActor in
                    self?.output.append(fragment)
                }
            }
            status = Self.finishedStatus(loadSeconds: loadedModelLoadSeconds, stats: stats)
        } catch {
            output = error.localizedDescription
            status = "Generation failed"
            engine = nil
            loadedSnapshotPath = nil
            loadedModelLoadSeconds = nil
        }
    }

    private static func finishedStatus(
        loadSeconds: TimeInterval?,
        stats: EreduGenerationStats
    ) -> String {
        let load = loadSeconds.map { String(format: "%.2fs", $0) } ?? "—"
        let ttft = stats.generatedTokens > 0
            ? String(format: "%.2fs", stats.timeToFirstToken)
            : "—"
        let throughput = stats.tokensPerSecond > 0
            ? String(format: "%.1f tok/s", stats.tokensPerSecond)
            : "—"
        return "Finished • load \(load) • TTFT \(ttft) • \(throughput)"
    }

    private func downloadSnapshot(_ repoID: Repo.ID) async throws -> URL {
        let entries = try await hub.listFiles(in: repoID, recursive: true)
            .filter { entry in
                entry.type == .file
                    && Self.downloadedFileExtensions.contains(
                        URL(fileURLWithPath: entry.path).pathExtension.lowercased()
                    )
            }
        let totalBytes = max(
            entries.reduce(Int64(0)) { $0 + Int64(max($1.size ?? 1, 1)) },
            1
        )
        var completedBytes: Int64 = 0

        for entry in entries {
            let fileBytes = Int64(max(entry.size ?? 1, 1))
            let completedBeforeFile = completedBytes
            let fileProgress = Progress(totalUnitCount: fileBytes)
            let samplingTask = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    let currentFileBytes = Int64(Double(fileBytes) * fileProgress.fractionCompleted)
                    self?.reportDownloadProgress(
                        completedBytes: completedBeforeFile + currentFileBytes,
                        totalBytes: totalBytes
                    )
                    try? await Task.sleep(for: .milliseconds(100))
                }
            }

            do {
                _ = try await hub.downloadFile(entry, from: repoID, progress: fileProgress)
            } catch {
                samplingTask.cancel()
                _ = await samplingTask.result
                throw error
            }
            samplingTask.cancel()
            _ = await samplingTask.result
            completedBytes += fileBytes
            reportDownloadProgress(completedBytes: completedBytes, totalBytes: totalBytes)
        }

        guard let revision = cache.resolveRevision(repo: repoID, kind: .model, ref: "main") else {
            throw CocoaError(.fileReadUnknown, userInfo: [
                NSLocalizedDescriptionKey: "The downloaded snapshot has no resolved revision"
            ])
        }
        return cache.snapshotsDirectory(repo: repoID, kind: .model)
            .appendingPathComponent(revision, isDirectory: true)
    }

    private func reportDownloadProgress(completedBytes: Int64, totalBytes: Int64) {
        let fraction = min(max(Double(completedBytes) / Double(totalBytes), 0), 1)
        downloadProgress = fraction
        status = "Downloading \(Int(fraction * 100))%"
    }

    private func snapshotURL(for model: CachedModel) -> URL {
        guard let repoID = Repo.ID(rawValue: model.repoID) else {
            return cache.cacheDirectory.appendingPathComponent("invalid-model")
        }
        return cache.snapshotsDirectory(repo: repoID, kind: .model)
            .appendingPathComponent(model.revision, isDirectory: true)
    }

    private func restoreModels() {
        var restored: [String: CachedModel] = [:]
        if let data = UserDefaults.standard.data(forKey: Self.recordsKey),
           let decoded = try? JSONDecoder().decode([CachedModel].self, from: data)
        {
            for model in decoded where isUsable(model) {
                restored[model.repoID] = model
            }
        }

        for model in discoverModels() where restored[model.repoID] == nil {
            restored[model.repoID] = model
        }

        models = restored.values.sorted {
            $0.repoID.localizedCaseInsensitiveCompare($1.repoID) == .orderedAscending
        }
        selectedModelID = models.first?.id
        saveModels()
    }

    private func discoverModels() -> [CachedModel] {
        let fileManager = FileManager.default
        guard let repoDirectories = try? fileManager.contentsOfDirectory(
            at: cache.cacheDirectory,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else { return [] }

        return repoDirectories.compactMap { repoDirectory in
            guard let repoID = Self.repoID(fromCacheDirectoryName: repoDirectory.lastPathComponent),
                  let parsedRepoID = Repo.ID(rawValue: repoID),
                  let revision = newestUsableRevision(for: parsedRepoID)
            else { return nil }
            return CachedModel(repoID: repoID, revision: revision)
        }
    }

    private func newestUsableRevision(for repoID: Repo.ID) -> String? {
        if let mainRevision = cache.resolveRevision(repo: repoID, kind: .model, ref: "main") {
            let mainModel = CachedModel(repoID: repoID.description, revision: mainRevision)
            if isUsable(mainModel) {
                return mainRevision
            }
        }

        let snapshotsDirectory = cache.snapshotsDirectory(repo: repoID, kind: .model)
        let keys: Set<URLResourceKey> = [.isDirectoryKey, .contentModificationDateKey]
        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDirectory,
            includingPropertiesForKeys: Array(keys),
            options: [.skipsHiddenFiles]
        ) else { return nil }

        return snapshots
            .filter {
                let model = CachedModel(repoID: repoID.description, revision: $0.lastPathComponent)
                return isUsable(model)
            }
            .sorted {
                let lhs = try? $0.resourceValues(forKeys: keys).contentModificationDate
                let rhs = try? $1.resourceValues(forKeys: keys).contentModificationDate
                return (lhs ?? .distantPast) > (rhs ?? .distantPast)
            }
            .first?
            .lastPathComponent
    }

    private func isUsable(_ model: CachedModel) -> Bool {
        let snapshot = snapshotURL(for: model)
        let fileManager = FileManager.default
        guard fileManager.fileExists(atPath: snapshot.appendingPathComponent("config.json").path),
              let enumerator = fileManager.enumerator(
                  at: snapshot,
                  includingPropertiesForKeys: nil,
                  options: [.skipsHiddenFiles]
              )
        else { return false }
        return enumerator.contains { item in
            (item as? URL)?.pathExtension.lowercased() == "safetensors"
        }
    }

    private func saveModels() {
        guard let data = try? JSONEncoder().encode(models) else { return }
        UserDefaults.standard.set(data, forKey: Self.recordsKey)
    }

    private static func repoID(fromCacheDirectoryName name: String) -> String? {
        let prefix = "models--"
        guard name.hasPrefix(prefix) else { return nil }
        let encoded = String(name.dropFirst(prefix.count))
        guard let separator = encoded.range(of: "--") else { return nil }
        return String(encoded[..<separator.lowerBound]) + "/" + String(encoded[separator.upperBound...])
    }

    private static func migrateLegacyCacheIfNeeded(to huggingFaceDirectory: URL) {
        let fileManager = FileManager.default
        let caches = fileManager.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        let legacyDirectory = caches.appendingPathComponent("huggingface", isDirectory: true)
        guard fileManager.fileExists(atPath: legacyDirectory.path),
              legacyDirectory.standardizedFileURL != huggingFaceDirectory.standardizedFileURL
        else { return }

        if !fileManager.fileExists(atPath: huggingFaceDirectory.path) {
            try? fileManager.createDirectory(
                at: huggingFaceDirectory.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            do {
                try fileManager.moveItem(at: legacyDirectory, to: huggingFaceDirectory)
                return
            } catch {
                // Fall through to the per-repository migration. A failed whole-tree
                // move must not hide a still-valid legacy cache from discovery.
            }
        }

        let legacyHub = legacyDirectory.appendingPathComponent("hub", isDirectory: true)
        let newHub = huggingFaceDirectory.appendingPathComponent("hub", isDirectory: true)
        try? fileManager.createDirectory(at: newHub, withIntermediateDirectories: true)
        guard let entries = try? fileManager.contentsOfDirectory(
            at: legacyHub,
            includingPropertiesForKeys: nil
        ) else { return }
        for source in entries {
            let destination = newHub.appendingPathComponent(source.lastPathComponent)
            guard !fileManager.fileExists(atPath: destination.path) else { continue }
            try? fileManager.moveItem(at: source, to: destination)
        }
    }
}
