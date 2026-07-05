import Testing
@testable import MLXUI

/// Covers the gated-model auth mapping (backlog T3): the HF API status → `InstallError`
/// translation that drives the `needsAuth` install state + Settings routing.
struct InstallAuthTests {
    private func isNeedsAuth(_ error: InstallError?) -> Bool {
        if case .needsAuth = error { return true }
        return false
    }

    @Test func unauthorizedMapsToNeedsAuth() {
        #expect(isNeedsAuth(InstallError.fromHTTPStatus(401, modelId: "x")))
        #expect(isNeedsAuth(InstallError.fromHTTPStatus(403, modelId: "x")))
    }

    @Test func notFoundMapsToModelNotFound() {
        if case .modelNotFound(let id)? = InstallError.fromHTTPStatus(404, modelId: "abc") {
            #expect(id == "abc")
        } else {
            Issue.record("Expected .modelNotFound for 404")
        }
    }

    @Test func successStatusesMapToNil() {
        #expect(InstallError.fromHTTPStatus(200, modelId: "x") == nil)
        #expect(InstallError.fromHTTPStatus(500, modelId: "x") == nil)
    }

    @Test func needsAuthStateIsNotActive() {
        #expect(InstallState.needsAuth("gated").isActive == false)
    }
}
