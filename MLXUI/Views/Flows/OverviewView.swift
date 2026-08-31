import SwiftUI

/// The Automate → "Overview" intro page: a long-form explainer for the readable
/// workflow concept (Chain Asset Transform, auto chaining, the DAG-as-a-list idea).
/// Pure marketing copy — no state, no interactivity.
struct OverviewView: View {
    private var contentWidth: CGFloat { 680 }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                //title
                whatIsIt
                background
                threeThings
            }
            .frame(maxWidth: contentWidth, alignment: .leading)
            .padding(32)
        }
        .navigationTitle("Overview")
    }

    private var title: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Overview")
                .font(.largeTitle.weight(.bold))
            Text("What mlx-workflow is, and why a workflow should read like a list.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .padding(.bottom, 28)
    }

    // MARK: - Background

    private var background: some View {
        VStack(alignment: .leading, spacing: 14) {
            header1("Why?")
            Text("\u{201C}AI is making software construction more accessible, but also making software harder to understand.\u{201D}")
                .font(.title3.weight(.medium))
                .italic()
        }
        .padding(.bottom, 28)
    }

    // MARK: - What is mlx-workflow?

    private var whatIsIt: some View {
        VStack(alignment: .leading, spacing: 14) {
            header1("What is mlx-workflow?")
            Text("A Readable AI Workflow you can Run locally")
                .font(.title3.weight(.semibold))
            //header1("A Readable AI Workflow you can Run locally")
            workflowCode("""
                1. Read Audio   client-call.m4a
                2. Transcribe   Whisper Large v3
                3. Summarize    Qwen3 8B; "Action items only"
                4. Save Text    followups.md
                """)
            Text("Note: 'Whisper Large v3' and 'Qwen3 8B' are AI models")
                .font(.callout)
                .foregroundStyle(.secondary)
            Text("That's an entire AI Workflow. You can read it, so you can change it. No developer, no canvas full of boxes and arrows, no wondering what it did while you weren't looking.")
            Text("It runs on your own machine\u{2014}we're currently on Mac\u{2014}so your client files never leave it, and running it costs you nothing.")
            Text("Keep using Claude or ChatGPT for the thinking and the one-off stuff. That's what they're for. When something becomes a weekly habit, move it here and it becomes a button you press.")
        }
        .padding(.bottom, 28)
    }

    // MARK: - 3 Things to Know

    private var threeThings: some View {
        VStack(alignment: .leading, spacing: 14) {
            header1("3 Things to Know")
            //Text("\u{201C}Your data. Your intelligence. Your workflow.\u{201D}")
            //    .font(.title3.weight(.medium))
            //    .italic()
            //    .padding(.bottom, 6)

            cat
            autoChaining
            dag
        }
    }

    private var cat: some View {
        VStack(alignment: .leading, spacing: 14) {
            header2("1. Chain Asset Transform (CAT)")
            Text("The whole idea fits in one line: Asset \u{2192} Transform \u{2192} Asset.")
            overviewImage("mlx-workflow")
            Text("You start with something (a file, some text, an audio clip). You put it through one step. You get something new out the other side. That's it. That's the atom.")
            Text("The trick is that the thing coming out is the same kind of thing as the thing going in, so you can just do it again:")
            workflowCode("""
                1. Read Audio   talk.m4a
                2. Transcribe   Whisper Large v3; lang=en
                3. Summarize    Qwen3 8B; "TL;DR in 3 bullets"
                4. Save Text    summary.md
                """)
            Text("Four rows, top to bottom. Audio in, text out, summary out, file on disk. You already understood it without anyone explaining it, which is the point. A workflow is a numbered list, and a numbered list is something anybody can read, edit, email, or paste into a chat.")
            Text("Each row is one transform. Stack the rows and you've got a program.")
        }
        .padding(.bottom, 28)
    }

    private var autoChaining: some View {
        VStack(alignment: .leading, spacing: 14) {
            header2("2. Auto chaining and Referencing")
            Text("Position is the wiring. By default every row eats whatever the row above it produced. No arrows, no drag-and-drop noodles, no connecting little dots. Row 3 gets row 2's output because it sits under row 2. That's why the flow above needs zero plumbing.")
            Text("But sometimes a row doesn't want the row above. So you point it somewhere else with a number in parens:")
            workflowCode("""
                1. Read Audio         podcast_fr.m4a
                2. Transcribe         Whisper Large v3; lang=fr
                3. Translate          Llama 3.1 8B; to=English
                4. Summarize          Ministral 3B; "3 key points"
                5. Speak              Kokoro 82M; af_heart
                6. Save Audio         recap-en.wav
                7. Save Text    (2)   transcript-fr.txt
                8. Save Text    (3)   transcript-en.txt
                """)
            Text("Row 7 says (2), meaning \u{201C}skip the row above, give me row 2's French transcript.\u{201D} Row 8 reaches back to row 3. One run, three outputs.")
            Text("And a row can pull from several rows at once when it needs more than one input:")
            workflowCode("""
                1. Read Video             intro.mp4
                2. Read Video             main.mp4
                3. Read Video             outro.mp4

                4. Join Video   (1,2,3)
                5. Save Video             final.mp4
                """)
            Text("(1,2,3) bundles three outputs into one input, in that order. Cap is four references per row, so a row never turns into spaghetti.")
            Text("The design rule underneath all this: the common case stays invisible, the exception announces itself. Most rows have no parens, so when you see parens you know something interesting is happening. A reference only reads a result. It never jumps control around.")
        }
        .padding(.bottom, 28)
    }

    private var dag: some View {
        VStack(alignment: .leading, spacing: 14) {
            header2("3. Full Directed Acyclic Graph, written as a list")
            Text("Here's the part people miss. Those two rules give you a full directed acyclic graph. So you get the expressiveness of a node canvas without the canvas.")
            overviewImage("dag")
            Text("Fan-out, fan-in, multiple roots, branches that run side by side. That's every shape a DAG can take.")
            Text("A Readable Workflow is to a node graph what Markdown is to HTML. Same structure underneath, but one of them you can read in a plain text file, diff in git, and paste into an email.")
        }
    }

    // MARK: - Shared pieces

    /// Header 1 — page-level section titles.
    private func header1(_ text: String) -> some View {
        Text(text)
            .font(.title.weight(.bold))
    }

    /// Header 2 — sub-titles beneath a Header 1.
    private func header2(_ text: String) -> some View {
        Text(text)
            .font(.title3.weight(.semibold))
    }

    /// A numbered workflow list, rendered as a monospaced code block.
    private func workflowCode(_ text: String) -> some View {
        Text(text)
            .font(.system(.body, design: .monospaced))
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
            .background(.quaternary.opacity(0.25), in: RoundedRectangle(cornerRadius: 10))
    }

    /// A bundled diagram (mlx-workflow.png / dag.png), centered within the column.
    private func overviewImage(_ name: String) -> some View {
        Group {
            if let url = Bundle.main.url(forResource: name, withExtension: "png"),
               let image = NSImage(contentsOf: url) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: contentWidth)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12)
                            .strokeBorder(.quaternary, lineWidth: 1)
                    }
            }
        }
        .frame(maxWidth: .infinity)
    }
}
