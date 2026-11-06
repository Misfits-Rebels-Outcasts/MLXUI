import Foundation
import MLX
import Testing
@testable import MLXUI

/// MC3-4/MC3-5 — shared access to the checked-in reference fixtures and to the installed checkpoint.
///
/// The real-weights suites are opt-in twice over (issue "Running the gated tests", the `SAMAudioRealLoadTests`
/// pattern): they need `MLXUI_TEST_DEPTH=1` **and** `MLXUI_TEST_DEPTH_DIR` pointing at an installed copy. A
/// stray install must never start a run by itself, and a skipped suite does not count as evidence.
enum DepthAnythingRealLoad {
    nonisolated static let env = ProcessInfo.processInfo.environment

    nonisolated static var optedIn: Bool {
        (env["MLXUI_TEST_DEPTH"] ?? env["TEST_RUNNER_MLXUI_TEST_DEPTH"]) == "1"
    }

    nonisolated static var directory: URL? {
        guard let raw = env["MLXUI_TEST_DEPTH_DIR"] ?? env["TEST_RUNNER_MLXUI_TEST_DEPTH_DIR"] else { return nil }
        let url = URL(fileURLWithPath: raw)
        let manager = FileManager.default
        guard manager.fileExists(atPath: url.appendingPathComponent("config.json").path),
              manager.fileExists(atPath: url.appendingPathComponent("model.safetensors").path) else { return nil }
        return url
    }

    nonisolated static var canRun: Bool { optedIn && directory != nil }
}

/// The checked-in `MLXUITests/Fixtures/DepthAnything` corpus (generated once by `make_depth_golden.py`; never
/// edited to make a change pass).
enum DepthAnythingFixtures {
    nonisolated static var directory: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()   // MLXUITests
            .appendingPathComponent("Fixtures/DepthAnything")
    }

    nonisolated static func golden() throws -> [String: Any] {
        let url = directory.appendingPathComponent("depth-golden.json")
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any] else {
            throw DepthAnythingError.badInput("depth-golden.json is not a JSON object")
        }
        return object
    }

    nonisolated static func caseInfo(_ name: String) throws -> [String: Any] {
        let golden = try golden()
        let cases = try #require(golden["cases"] as? [String: Any])
        return try #require(cases[name] as? [String: Any])
    }

    /// Walk into a golden.json node by keys; a key that parses as an Int indexes an array.
    nonisolated static func node(_ root: Any, _ keys: [String]) throws -> Any {
        var node: Any = root
        for key in keys {
            if let dictionary = node as? [String: Any], let next = dictionary[key] {
                node = next
            } else if let array = node as? [Any], let index = Int(key), array.indices.contains(index) {
                node = array[index]
            } else {
                throw DepthAnythingError.badInput("fixture path \(keys) missing")
            }
        }
        return node
    }

    nonisolated static func shape(_ name: String, _ keys: [String]) throws -> [Int] {
        let node = try node(try caseInfo(name), keys)
        let shapeObject = (node as? [String: Any])?["shape"] ?? node
        return try #require(shapeObject as? [Int])
    }

    nonisolated static func floats(_ file: String) throws -> [Float] {
        let data = try Data(contentsOf: directory.appendingPathComponent(file))
        return data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    nonisolated static func bytes(_ file: String) throws -> [UInt8] {
        try [UInt8](Data(contentsOf: directory.appendingPathComponent(file)))
    }

    nonisolated static func array(_ file: String, shape: [Int]) throws -> MLXArray {
        let values = try floats(file)
        let count = shape.reduce(1, *)
        #expect(values.count == count, "\(file) has \(values.count) floats, expected \(count)")
        return MLXArray(values).reshaped(shape)
    }

    /// The reference's NHWC preprocessed tensor for a case.
    nonisolated static func preprocessed(_ caseName: String) throws -> MLXArray {
        let shape = try shape(caseName, ["preprocessed"])
        let file = try file(for: caseName, ["preprocessed"])
        return try array(file, shape: shape)
    }

    nonisolated static func file(for caseName: String, _ keys: [String]) throws -> String {
        let node = try node(try caseInfo(caseName), keys)
        return try #require((node as? [String: Any])?["file"] as? String)
    }

    /// Append one evidence line to the test host's temporary directory, so the PR's parity numbers can be collected
    /// from a plain run (the sandboxed host cannot write the caller's folder; the standard output is captured by
    /// Xcode rather than relayed).
    nonisolated static func record(_ line: String) {
        guard let data = (line + "\n").data(using: .utf8) else { return }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("DEPTH-PARITY.txt")
        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(data)
            try? handle.close()
        } else {
            try? data.write(to: url)
        }
    }

    /// Max / p99 / median absolute difference against a raw fixture.
    nonisolated static func differenceStats(_ array: MLXArray, file: String) throws -> (max: Float, p99: Float, median: Float) {
        let actual = try MLXErrorGuard.read(array.asType(.float32), as: Float.self)
        let reference = try floats(file)
        guard actual.count == reference.count else {
            throw DepthAnythingError.badInput("\(file): \(actual.count) values vs \(reference.count)")
        }
        let deltas = zip(actual, reference).map { abs($0 - $1) }.sorted()
        guard !deltas.isEmpty else { throw DepthAnythingError.badInput("\(file) is empty") }
        return (deltas.last ?? .infinity,
                deltas[min(deltas.count - 1, Int(Double(deltas.count - 1) * 0.99))],
                deltas[deltas.count / 2])
    }

    /// Maximum absolute difference between an array and a raw little-endian fixture, printing it RIFE-style.
    nonisolated static func maxDifference(_ array: MLXArray, file: String) throws -> Float {
        let actual = try MLXErrorGuard.read(array.asType(.float32), as: Float.self)
        let reference = try floats(file)
        guard actual.count == reference.count else {
            throw DepthAnythingError.badInput("\(file): \(actual.count) values vs \(reference.count)")
        }
        return zip(actual, reference).map { abs($0 - $1) }.max() ?? .infinity
    }
}
