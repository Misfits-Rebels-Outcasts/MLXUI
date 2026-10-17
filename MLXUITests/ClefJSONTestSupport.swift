import Foundation

@testable import MLXUI

/// Reading a generated golden through the port's own parser keeps the fixture's number kinds (`int`
/// vs `float`) and object order intact — which is exactly what the port must reproduce. Shared by
/// `ClefGoldenTests` (CF-5a) and `ClefParityTests` (CF-5b).
extension ClefJSON {
    subscript(key: String) -> ClefJSON? {
        guard case .object(let fields) = self else { return nil }
        return fields.first { $0.key == key }?.value
    }

    var arrayValue: [ClefJSON]? {
        guard case .array(let items) = self else { return nil }
        return items
    }

    var dictionaryValue: [String: ClefJSON]? {
        guard case .object(let fields) = self else { return nil }
        return Dictionary(uniqueKeysWithValues: fields.map { ($0.key, $0.value) })
    }

    var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    var intValue: Int? {
        guard case .int(let value) = self else { return nil }
        return value
    }

    var doubleValue: Double? {
        switch self {
        case .double(let value): return value
        case .int(let value): return Double(value)
        default: return nil
        }
    }

    var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    var intArray: [Int]? {
        arrayValue?.compactMap(\.intValue)
    }

    var spanValue: Range<Int>? {
        guard let pair = intArray, pair.count == 2 else { return nil }
        return pair[0] ..< pair[1]
    }
}

