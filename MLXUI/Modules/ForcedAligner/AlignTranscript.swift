import Foundation

/// The transcript of the alignment run in progress: **run input**, not a load setting, so it is neither in `StageConfig`
/// nor in the engine-cache key. The caller sets it around `stage.run`
/// (`AlignTranscript.$current.withValue(text) { … }`, done by `RealExecutor.runAlign` and `AlignRunModel.start`), and
/// `ForcedAlignerStage.run` reads it. It follows the task through the lease wrappers, which a new unstructured or
/// detached task between the two would break.
///
/// Owner ruling 17 (Boo, 2026-11-27): "The transcript must not be part of the engine cache key. It's run input, not a
/// load setting. Find where the flow row's key picks it up. If it comes from StageConfig.transcript feeding a key built
/// from the whole config, the preferred fix is to pass the transcript to the stage as run input and remove
/// StageConfig.transcript. That supersedes ruling 5, so update the CLAUDE.md note from 9492f47 to match. If the fix
/// stays inside the aligner's engine, stage, SDK, module and RealExecutor's .align branch, do it in this PR. Add a test
/// that a flow aligning two different transcripts with the same language makes exactly one cache entry and one booking.
/// If it needs changes to EngineCache, MemoryBudget or how keys are built for every stage, stop at point A with a
/// recommendation."
///
/// Owner ruling 19 (Boo, 2026-11-27): "Before using a task-local value, check whether RealExecutor's .align branch and
/// the Run screen can take the stage the cache returns, cast it to ForcedAlignerStage, and call a method with the
/// transcript as an explicit argument (for example align(audio:transcript:language:)). That keeps PipelineStage, Media
/// and EngineCache unchanged. A failed cast is a clear error, never a silent fallback. If that works, use it. If it
/// doesn't, use the task-local value with both of the lead's guards, and say in the commit message why the explicit
/// argument didn't work. Either way, StageConfig.transcript goes, and the CLAUDE.md note is updated to match."
///
/// The explicit argument did not work: the cache returns a lease wrapper, not a `ForcedAlignerStage`. A flow row gets a
/// `LeasedStage`; the Run screen runs a `LeasingStage`, which takes its hold inside `run` and keeps the stage private,
/// so calling the aligner with an argument would run it outside the lease unless `EngineCache.swift` changed.
nonisolated enum AlignTranscript {
    @TaskLocal static var current: String?

    /// The stage ran with no transcript set. Never answered with an empty or an earlier one.
    struct Missing: LocalizedError, Equatable {
        var errorDescription: String? {
            "The aligner got no transcript for this run — run Align again with the text of the recording."
        }
    }
}
