import Testing
import Foundation
@testable import MLXUI

// S1-5f — Apple guideline 5.1.2(i): explicit consent before a third-party provider key is saved.

@MainActor
struct ProviderConsentTests {
    @Test func bothKindsNameTheProviderAndSayHowToWithdraw() {
        for (kind, provider) in [(ProvidersSettingsView.Kind.provider, "anthropic"), (.search, "tavily")] {
            let sentence = ProvidersSettingsView.consentSentence(kind: kind, provider: provider)
            #expect(sentence.contains(provider.capitalized))
            #expect(sentence.contains("Remove the key"))
        }
    }

    @Test func theTwoKindsSayWhatIsSent() {
        #expect(ProvidersSettingsView.consentSentence(kind: .provider, provider: "openai").contains("text from your documents"))
        #expect(ProvidersSettingsView.consentSentence(kind: .search, provider: "brave").contains("search text"))
    }

    /// The HuggingFace token sheet (ModelsSettingsView) passes no `consent:`, so it keeps the plain
    /// Save button. Built here exactly as that call site builds it.
    @Test func theHuggingFaceTokenSheetHasNoConsent() {
        let sheet = CredentialKeySheet(
            title: "HuggingFace Access Token",
            choices: ["huggingface"],
            placeholder: { _ in "Paste your HuggingFace token" },
            isValid: { HFTokenValidator.isPlausible($0) },
            signupLink: { _ in nil },
            onSave: { _, _ in })
        #expect(sheet.consent == nil)
    }

    @Test func aProviderSheetCarriesTheConsentForTheSelectedProvider() {
        let sheet = CredentialKeySheet(
            title: "Add provider key", choices: ["groq"], placeholder: { _ in "" },
            signupLink: { _ in nil },
            consent: { ProvidersSettingsView.consentSentence(kind: .provider, provider: $0) },
            onSave: { _, _ in })
        #expect(sheet.consent?("groq").contains("Groq") == true)
    }
}
