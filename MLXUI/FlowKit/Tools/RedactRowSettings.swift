import Foundation

/// SPEC-Q252 (PF) — the `Redact` row's settings parse, shared with the validator's check 25 (E725). The port of
/// `catflow-mlx/src/catflow/tools/redact_row.py::parse_redact`: one setting, `only=`, a comma-separated subset of the eight
/// writable category names; absent means all eight. Any other key, any bare word, an empty category and an unknown category
/// are refused by name — never ignored.
nonisolated struct RedactSettingsError: Error, Equatable {
    /// Completes "Row N's `Redact` ..." (E725).
    var detail: String
}

nonisolated enum RedactRowSettings {
    private static let list = "person, email, phone, url, address, date, account and secret"

    /// The categories `only=` selects, in the canonical order; `nil` = all eight.
    static func parse(_ raw: String?) throws -> [PrivacyFilterCategory]? {
        let s = FlowSettings(raw)
        if let word = s.firstBare() {
            throw RedactSettingsError(detail: "takes only `only=` (a comma-separated subset of \(list)); remove `\(word)`")
        }
        for key in s.orderedKeys where key != "only" {
            throw RedactSettingsError(detail: "takes only `only=` (a comma-separated subset of \(list)); remove `\(key)=`")
        }
        guard let value = s.value(for: "only") else { return nil }
        var chosen: Set<PrivacyFilterCategory> = []
        for token in value.split(separator: ",", omittingEmptySubsequences: false) {
            let name = token.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if name.isEmpty {
                throw RedactSettingsError(detail: "has an empty category in `only=` -- use a comma-separated subset of \(list)")
            }
            guard let category = PrivacyFilterCategory(rawValue: name) else {
                throw RedactSettingsError(detail: "doesn't know the category \"\(name)\" -- use a comma-separated subset of \(list)")
            }
            chosen.insert(category)
        }
        return PrivacyFilterCategory.allCases.filter { chosen.contains($0) }
    }
}
