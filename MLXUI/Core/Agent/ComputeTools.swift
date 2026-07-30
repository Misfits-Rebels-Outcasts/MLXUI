//
//  ComputeTools.swift
//  MLXUI — agentic chat tools, slice AG3.
//
//  In-process compute tools that need no sandbox change:
//    • run_javascript  — JavaScriptCore eval (the safe "shell" stand-in: no FS/network bridge)
//    • calculator      — arithmetic only; charset-guarded then evaluated on the JS path
//    • datetime        — the current date/time (UTC ISO-8601 + local)
//    • read_clipboard  — reads the pasteboard text (read-only)
//    • write_clipboard — writes the pasteboard text (**side-effecting ⇒ requiresApproval**)
//
//  As in AG2, the risky/pure logic lives in `static` helpers exercised by synchronous unit
//  tests; `execute` is a thin wrapper. (Sync suites run reliably; see the AG0-b anomaly.)
//

import Foundation
import JavaScriptCore
import AppKit
import MLXLMCommon

/// Evaluates a JavaScript snippet in an isolated `JSContext` and returns its result as text.
/// No host objects, filesystem, or network are exposed — a bare `JSContext` has only the JS
/// standard library — so this is the sandbox-safe stand-in for a shell.
nonisolated struct RunJavaScriptTool: AgentTool {
    let name = "run_javascript"
    let toolDescription = """
        Evaluate a JavaScript snippet and return its result. Runs in an isolated interpreter \
        with only the JavaScript standard library (Math, JSON, string/array methods, etc.) — \
        no file, network, or system access. Use the last expression's value as the result.
        """
    let parameters: [ToolParameter] = [
        .required("script", type: .string, description: "The JavaScript source to evaluate.")
    ]

    // JavaScriptCore's public API exposes no execution-time limit (the C
    // `JSContextGroupSetExecutionTimeLimit` lives in a private header), so a runaway script
    // (e.g. `while (true) {}`) can't be interrupted *inside* the tool. AG6's dispatch-level
    // timeout bounds it instead: after this many seconds the agent loop abandons the call and
    // reports a timeout to the model. Caveat — the JS keeps spinning on its background task
    // until the process exits (cancellation-deaf); this rail unblocks the loop, not the thread.
    var executionTimeout: TimeInterval? { 15 }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        Self.evaluate(arguments.string("script") ?? "")
    }

    /// Pure evaluator: returns the result string, a `JavaScript error: …`, or a diagnostic —
    /// never throws.
    static func evaluate(_ script: String) -> String {
        let trimmed = script.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Error: missing 'script' argument." }
        guard let context = JSContext() else { return "Error: could not create a JavaScript context." }

        var thrown: String?
        context.exceptionHandler = { _, exception in
            thrown = exception?.toString() ?? "unknown JavaScript error"
        }

        let result = context.evaluateScript(script)
        if let thrown { return "JavaScript error: \(thrown)" }
        guard let result, !result.isUndefined else { return "undefined" }
        if result.isNull { return "null" }
        return result.toString() ?? "undefined"
    }
}

/// Evaluates a pure arithmetic expression. Input is restricted to numbers and arithmetic
/// operators, then evaluated on the JS path — so there is no way to reach identifiers or host
/// objects.
nonisolated struct CalculatorTool: AgentTool {
    let name = "calculator"
    let toolDescription = """
        Evaluate an arithmetic expression and return the numeric result. Supports + - * / % \
        parentheses and decimals (e.g. "3 + 4 * (2 - 1)").
        """
    let parameters: [ToolParameter] = [
        .required("expression", type: .string, description: "The arithmetic expression to evaluate.")
    ]

    /// Characters permitted in an expression — digits, arithmetic operators, grouping, decimal
    /// point, exponent marker, and whitespace. No letters (beyond `e`/`E`) ⇒ no identifiers.
    static let allowed = CharacterSet(charactersIn: "0123456789+-*/%(). eE\t")

    func execute(arguments: [String: JSONValue]) async throws -> String {
        Self.evaluate(arguments.string("expression") ?? "")
    }

    /// Pure evaluator: charset-guards, then defers to the JS evaluator; never throws.
    static func evaluate(_ expression: String) -> String {
        let trimmed = expression.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return "Error: missing 'expression' argument." }
        guard trimmed.unicodeScalars.allSatisfy({ allowed.contains($0) }) else {
            return "Error: calculator accepts only numbers and the + - * / % ( ) . operators."
        }
        return RunJavaScriptTool.evaluate("(\(trimmed))")
    }
}

/// Returns the current date and time.
nonisolated struct DateTimeTool: AgentTool {
    let name = "datetime"
    let toolDescription = "Return the current date and time (UTC ISO-8601 and the user's local time)."
    let parameters: [ToolParameter] = []

    func execute(arguments: [String: JSONValue]) async throws -> String {
        Self.format(Date())
    }

    /// Pure formatter for a given instant — testable without reading the wall clock.
    static func format(_ date: Date) -> String {
        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime]
        let utc = iso.string(from: date)
        let local = date.formatted(date: .abbreviated, time: .standard)
        return "UTC (ISO 8601): \(utc)\nLocal (\(TimeZone.current.identifier)): \(local)"
    }
}

/// Reads the current text contents of the system pasteboard (read-only).
nonisolated struct ReadClipboardTool: AgentTool {
    let name = "read_clipboard"
    let toolDescription = "Read the current text contents of the clipboard."
    let parameters: [ToolParameter] = []

    func execute(arguments: [String: JSONValue]) async throws -> String {
        Self.describe(NSPasteboard.general.string(forType: .string))
    }

    /// Pure presenter — separates pasteboard I/O from formatting for testing.
    static func describe(_ contents: String?) -> String {
        guard let contents, !contents.isEmpty else { return "The clipboard is empty (no text)." }
        return contents
    }
}

/// Writes text to the system pasteboard. Side-effecting ⇒ gated behind user approval.
nonisolated struct WriteClipboardTool: AgentTool {
    let name = "write_clipboard"
    let toolDescription = "Copy the given text to the clipboard, replacing its current contents."
    let parameters: [ToolParameter] = [
        .required("text", type: .string, description: "The text to place on the clipboard.")
    ]
    var requiresApproval: Bool { true }

    func execute(arguments: [String: JSONValue]) async throws -> String {
        let text = arguments.string("text") ?? ""
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
        return "Copied \(text.count) character\(text.count == 1 ? "" : "s") to the clipboard."
    }
}
