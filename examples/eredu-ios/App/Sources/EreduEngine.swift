import Foundation

enum EreduEngineError: LocalizedError {
    case native(String)
    case missingMetalLibrary

    var errorDescription: String? {
        switch self {
        case .native(let message): message
        case .missingMetalLibrary: "mlx.metallib is missing from the application bundle"
        }
    }
}

private final class StreamContext: @unchecked Sendable {
    let receive: @Sendable (String) -> Void

    init(receive: @escaping @Sendable (String) -> Void) {
        self.receive = receive
    }
}

private func receiveText(
    _ bytes: UnsafePointer<UInt8>?,
    _ length: Int,
    _ context: UnsafeMutableRawPointer?
) {
    guard let bytes, let context else { return }
    let sink = Unmanaged<StreamContext>.fromOpaque(context).takeUnretainedValue()
    sink.receive(String(decoding: UnsafeBufferPointer(start: bytes, count: length), as: UTF8.self))
}

private func consumeNativeError(_ pointer: UnsafeMutablePointer<CChar>?) -> String {
    guard let pointer else { return "unknown Eredu error" }
    defer { eredu_string_free(pointer) }
    return String(cString: pointer)
}

struct EreduGenerationStats: Sendable {
    let generatedTokens: UInt64
    let timeToFirstToken: TimeInterval
    let tokensPerSecond: Double
}

/// Swift owner for the opaque Rust worker. The worker keeps all MLX objects on
/// one native thread even when Swift tasks resume on different executors.
final class EreduEngine: @unchecked Sendable {
    private let handle: OpaquePointer

    private init(handle: OpaquePointer) {
        self.handle = handle
    }

    deinit {
        eredu_model_free(handle)
    }

    static func load(modelAt modelURL: URL) async throws -> EreduEngine {
        guard let metallibURL = Bundle.main.url(forResource: "mlx", withExtension: "metallib") else {
            throw EreduEngineError.missingMetalLibrary
        }
        return try await Task.detached(priority: .userInitiated) {
            var nativeError: UnsafeMutablePointer<CChar>?
            let handle = modelURL.path.withCString { modelPath in
                metallibURL.path.withCString { metallibPath in
                    eredu_model_create(modelPath, metallibPath, &nativeError)
                }
            }
            guard let handle else {
                throw EreduEngineError.native(consumeNativeError(nativeError))
            }
            return EreduEngine(handle: handle)
        }.value
    }

    func generate(
        prompt: String,
        onText: @escaping @Sendable (String) -> Void
    ) async throws -> EreduGenerationStats {
        try await Task.detached(priority: .userInitiated) { [self] in
            let streamContext = Unmanaged.passRetained(StreamContext(receive: onText))
            defer { streamContext.release() }
            var nativeError: UnsafeMutablePointer<CChar>?
            var generatedTokens: UInt64 = 0
            var timeToFirstToken = 0.0
            var tokensPerSecond = 0.0
            let status = prompt.withCString { promptPointer in
                eredu_model_generate(
                    handle,
                    promptPointer,
                    receiveText,
                    streamContext.toOpaque(),
                    &generatedTokens,
                    &timeToFirstToken,
                    &tokensPerSecond,
                    &nativeError
                )
            }
            guard status == 0 else {
                throw EreduEngineError.native(consumeNativeError(nativeError))
            }
            return EreduGenerationStats(
                generatedTokens: generatedTokens,
                timeToFirstToken: timeToFirstToken,
                tokensPerSecond: tokensPerSecond
            )
        }.value
    }
}
