import Foundation

/// FV-2 (`RSI/DelegateFrameViewBacklog.md` §4.1) — the single renderer a real run and the
/// read-only Properties-tab preview both call, so the two cannot diverge. Before this, the
/// choice of *which* renderer a `.frame` task gets was decided twice: once inline in
/// `RealExecutor.runModel` (`FrameRenderer.render`, the generator path) and once in
/// `RealExecutor.deciderPrompt` (`DeciderFrame.renderJudgeFrame` / `renderThinkFrame` /
/// `renderFrame`, chosen by task name — fact 7). This is that decision, made once: whether
/// `task` is one of `TaskCatalog.deciderTasks` picks the family, and within deciders, the task
/// name picks Judge / Think / everything else — exactly `RealExecutor.deciderPrompt`'s own
/// branching, ported here so both call sites share it.
nonisolated enum FramePreview {

    /// `refName` stripped to the bundle lookup name `FrameRenderer.loadFrame` takes — the
    /// `"frames/"` prefix and `".frame.txt"` suffix removed. Shared so "which file" is computed
    /// in exactly one place.
    static func frameFileName(refName: String) -> String {
        refName
            .replacingOccurrences(of: "frames/", with: "")
            .replacingOccurrences(of: ".frame.txt", with: "")
    }

    /// Renders the prompt `task`'s frame produces. `tags`/`tools`/`transcript`/`context` are
    /// only consumed by the branches that read them (`{tags}` outside Judge/Think, `{tools}`/
    /// `{transcript}` only inside Think, `context` only when the row splices `; ctx`) — passing
    /// them unconditionally for a task that ignores them is harmless, matching what
    /// `deciderPrompt` already did for its own Judge/Think split.
    ///
    /// `assets` carries whatever text each branch actually reads: real bound text for a run,
    /// or FV-2's edit-time stand-in labels for a preview — this function has no way to tell
    /// the two apart, by design (see `RSI/DelegateFrameViewBacklog.md` §6 "drift between
    /// preview and run").
    static func render(
        task: String,
        refName: String,
        settings: String?,
        assets: [Asset],
        tags: [String],
        tools: [String] = [],
        transcript: String = "(nothing yet)",
        context: String? = nil
    ) throws -> String {
        guard refName.hasPrefix("frames/") else {
            // FV-4-2: a task with no published frame at all is not a missing-file problem —
            // `missingFrame`'s "reinstall to restore it" would be the wrong fix to suggest.
            throw FrameError.noPublishedFrame(task: task)
        }
        let frameText = try FrameRenderer.loadFrame(named: frameFileName(refName: refName))

        guard TaskCatalog.deciderTasks[task] != nil else {
            return try FrameRenderer.render(frameText: frameText, settings: settings,
                                            assets: assets, tags: tags.isEmpty ? nil : tags)
        }
        if task == "Judge" {
            return DeciderFrame.renderJudgeFrame(frameText: frameText, settings: settings,
                                                 inputs: assets, tags: tags, context: context)
        }
        if task == "Think" {
            return DeciderFrame.renderThinkFrame(frameText: frameText, settings: settings,
                                                 inputs: assets, tags: tags, tools: tools,
                                                 transcript: transcript, context: context)
        }
        return DeciderFrame.renderFrame(frameText: frameText, settings: settings,
                                        inputs: assets, tags: tags, context: context)
    }
}
