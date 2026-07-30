import Foundation
import Testing
import MLXLMCommon
@testable import MLXUI

/// AG3 — compute tools (`run_javascript`, `calculator`, `datetime`, clipboard). The pure static
/// helpers hold the logic and are tested synchronously; the runaway-loop time limit and live
/// pasteboard I/O aren't exercised here (slow / side-effecting).
struct ComputeToolsTests {

    // MARK: run_javascript

    @Test func jsEvaluatesExpressions() {
        #expect(RunJavaScriptTool.evaluate("1 + 2") == "3")
        #expect(RunJavaScriptTool.evaluate("[1,2,3].map(x => x * 2).join(',')") == "2,4,6")
        #expect(RunJavaScriptTool.evaluate("JSON.stringify({a: 1})") == "{\"a\":1}")
    }

    @Test func jsReportsErrorsGracefully() {
        let out = RunJavaScriptTool.evaluate("throw new Error('boom')")
        #expect(out.contains("boom"))
        #expect(out.lowercased().contains("error"))
    }

    @Test func jsEmptyScriptIsRejected() {
        #expect(RunJavaScriptTool.evaluate("   ") == "Error: missing 'script' argument.")
    }

    @Test func jsUndefinedAndNull() {
        #expect(RunJavaScriptTool.evaluate("undefined") == "undefined")
        #expect(RunJavaScriptTool.evaluate("null") == "null")
    }

    @Test func jsHasNoHostBridge() {
        // A bare JSContext exposes no `require`, no filesystem, no networking.
        let out = RunJavaScriptTool.evaluate("typeof require")
        #expect(out == "undefined")
    }

    // MARK: calculator

    @Test func calculatorEvaluatesArithmetic() {
        #expect(CalculatorTool.evaluate("3 + 4 * (2 - 1)") == "7")
        #expect(CalculatorTool.evaluate("10 / 4") == "2.5")
        #expect(CalculatorTool.evaluate("10 % 3") == "1")
    }

    @Test func calculatorRejectsNonArithmetic() {
        // Letters (identifiers/function calls) are refused before evaluation.
        #expect(CalculatorTool.evaluate("Math.random()").hasPrefix("Error:"))
        #expect(CalculatorTool.evaluate("alert(1)").hasPrefix("Error:"))
    }

    @Test func calculatorEmptyIsRejected() {
        #expect(CalculatorTool.evaluate("").hasPrefix("Error:"))
    }

    // MARK: datetime

    @Test func datetimeFormatsUTCDeterministically() {
        let epoch = Date(timeIntervalSince1970: 0)
        let out = DateTimeTool.format(epoch)
        #expect(out.contains("1970-01-01T00:00:00Z"))
        #expect(out.contains("Local"))
    }

    // MARK: clipboard

    @Test func readClipboardDescribesEmptyAndPresent() {
        #expect(ReadClipboardTool.describe(nil) == "The clipboard is empty (no text).")
        #expect(ReadClipboardTool.describe("") == "The clipboard is empty (no text).")
        #expect(ReadClipboardTool.describe("hello") == "hello")
    }

    @Test func writeClipboardRequiresApproval() {
        #expect(WriteClipboardTool().requiresApproval == true)
        #expect(ReadClipboardTool().requiresApproval == false)
        #expect(RunJavaScriptTool().requiresApproval == false)
        #expect(CalculatorTool().requiresApproval == false)
    }

    @Test func computeToolsAreSpecced() {
        let fn = RunJavaScriptTool().toolSpec["function"] as? [String: any Sendable]
        let params = fn?["parameters"] as? [String: any Sendable]
        #expect(params?["required"] as? [String] == ["script"])
        #expect(DateTimeTool().parameters.isEmpty)
    }
}
