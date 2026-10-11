import Testing
import Foundation
@testable import MLXUI

/// R63 — a loaded model's weights are booked at their real size (safetensors on disk × the measured load factor),
/// by one function that every LLM load goes through.
struct ModelWeightChargeTests {

    private static let gb = Int64(1_073_741_824)
    private struct Sentinel: Error {}

    private static func modelDir(safetensors sizes: [Int], extra: Int = 4_096) throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("charge-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for (i, size) in sizes.enumerated() {
            let url = dir.appendingPathComponent("model-\(i).safetensors")
            FileManager.default.createFile(atPath: url.path, contents: nil)
            let handle = try FileHandle(forWritingTo: url)
            try handle.truncate(atOffset: UInt64(size))                       // sparse: no real disk used
            try handle.close()
        }
        try Data(count: extra).write(to: dir.appendingPathComponent("tokenizer.json"))   // never counted
        return dir
    }

    // MARK: - The function

    @Test func theChargeIsTheSafetensorsOnDiskTimesTheLoadFactor() throws {
        let dir = try Self.modelDir(safetensors: [6_000_000, 4_000_000])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelWeightCharge.safetensorsBytes(in: dir) == 10_000_000)
        #expect(ModelWeightCharge.bytes(directory: dir, catalogFallback: 9 * Self.gb) == 11_000_000)   // ×1.10, not the catalog
        #expect(ModelWeightCharge.bytes(directory: dir, catalogFallback: nil) == 11_000_000)
    }

    @Test func theLoadFactorNeverChargesLessThanTheFiles() {
        #expect(ModelWeightCharge.loadFactor >= 1.0)
        #expect(ModelWeightCharge.charge(onDiskBytes: 1_000) >= 1_000)
    }

    @Test func noSafetensorsFallsBackToTheCatalogFigureThenTheDirectorySize() throws {
        let dir = try Self.modelDir(safetensors: [], extra: 5_000)
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(ModelWeightCharge.bytes(directory: dir, catalogFallback: 7 * Self.gb) == 7 * Self.gb)   // today's conservative figure
        #expect(ModelWeightCharge.bytes(directory: dir, catalogFallback: nil) == 5_000)                   // as before: whole directory
        #expect(ModelWeightCharge.bytes(directory: dir.appendingPathComponent("missing"), catalogFallback: 3 * Self.gb) == 3 * Self.gb)
    }

    @Test func admissionAndTheLedgerShareOneBasis() throws {
        let dir = try Self.modelDir(safetensors: [8_000_000])
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(MLXChatBackend.weightsBytes(in: dir) == ModelWeightCharge.safetensorsBytes(in: dir))
        #expect(ModelWeightCharge.bytes(directory: dir, catalogFallback: nil)
                == ModelWeightCharge.charge(onDiskBytes: MLXChatBackend.weightsBytes(in: dir)))
    }

    // MARK: - Every caller goes through it

    /// The ledger total *inside the loader*, i.e. right after the pool booked the model.
    private final class Booked: @unchecked Sendable {
        private let lock = NSLock(); private var value: Int64 = -1
        func set(_ v: Int64) { lock.lock(); value = v; lock.unlock() }
        var bytes: Int64 { lock.lock(); defer { lock.unlock() }; return value }
    }

    private static func booked(by call: (ModelContainerPool, URL) async throws -> Void, dir: URL) async -> Int64 {
        let seen = Booked()
        let box = Box()
        let pool = ModelContainerPool(budgetBytes: 40 * gb) { _, _ in
            seen.set(box.pool?.budget.usedBytes ?? -2)
            throw Sentinel()
        }
        box.pool = pool
        try? await call(pool, dir)
        return seen.bytes
    }
    private final class Box: @unchecked Sendable { var pool: ModelContainerPool? }

    @Test func everyEntryPointBooksTheMeasuredChargeNotTheCatalogFigure() async throws {
        let dir = try Self.modelDir(safetensors: [5_000_000])      // 5 MB on disk
        defer { try? FileManager.default.removeItem(at: dir) }
        let expected = ModelWeightCharge.charge(onDiskBytes: 5_000_000)
        let model = makeEntry(id: "m", modelType: .llm, ramGB: 9)              // catalog 9 GiB: must not be what is booked

        #expect(await Self.booked(by: { pool, d in _ = try await pool.container(for: model, directory: d) }, dir: dir) == expected)
        #expect(await Self.booked(by: { pool, d in _ = try await pool.container(directory: d, footprintBytes: 9 * Self.gb) }, dir: dir) == expected)
        #expect(await Self.booked(by: { pool, d in _ = try await pool.container(directory: d) }, dir: dir) == expected)
        #expect(await Self.booked(by: { pool, d in
            _ = try await LLMEngine.generate(prompt: "hi", modelDir: d, maxTokens: 1, footprintBytes: 9 * Self.gb, pool: pool)
        }, dir: dir) == expected)
        // The server's backend asks the pool exactly as the others do (same call, its catalog figure as fallback).
        let seen = Booked(); let box = Box()
        let pool = ModelContainerPool(budgetBytes: 40 * Self.gb) { _, _ in seen.set(box.pool?.budget.usedBytes ?? -2); throw Sentinel() }
        box.pool = pool
        let backend = MLXChatBackend(directory: dir, footprintBytes: 9 * Self.gb, pool: pool)
        do { for try await _ in backend.generate(ServeRequest(model: "m", messages: [ServeMessage(role: .user, text: "hi")])) {} } catch {}
        #expect(seen.bytes == expected)
    }

    @Test func theLedgerTotalAfterTwoLoadsIsTheSumOfTheirCharges() async throws {
        let a = try Self.modelDir(safetensors: [3_000_000])
        let b = try Self.modelDir(safetensors: [5_000_000, 1_000_000])
        defer { try? FileManager.default.removeItem(at: a); try? FileManager.default.removeItem(at: b) }
        let budget = MemoryBudget(capacityBytes: 40 * Self.gb)
        let pool = ResidentPool<Int>(budget: budget)
        for dir in [a, b] {
            let bytes = ModelWeightCharge.bytes(directory: dir, catalogFallback: 9 * Self.gb)
            _ = try await pool.value(for: dir.path, bytes: bytes) { 1 }
        }
        #expect(budget.usedBytes == ModelWeightCharge.charge(onDiskBytes: 3_000_000) + ModelWeightCharge.charge(onDiskBytes: 6_000_000))
        #expect(budget.usedBytes == 9_900_000)
    }
}
