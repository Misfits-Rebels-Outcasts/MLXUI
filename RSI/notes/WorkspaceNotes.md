I'll read the actual workspace code to answer this accurately.Read the actual code. Here's what it does, in plain English — nothing changed.

## 1. Yes — Build runs `Ingest.cat`, Ask runs `DocChat.cat`

But not because anyone wired those two names together. The app works it out by **reading the flows**.

`WorkspaceKnowledge.swift` parses every `.cat` in the workspace and asks two questions of each:

- Does it have a `Store Index <name>` row? → it's a **builder** for that index.
- Does it have a `Read Index <name>` row *and* a `Retrieve` (or `Keyword Search`) row somewhere? → it's a **querier** of that index.

In `ask_your_docs`:
- `Ingest.cat` row 6 is `Store Index library.index` → builder
- `DocChat.cat` row 1 is `Read Index library.index`, row 5 is `Retrieve` → querier

Same index name on both sides, exactly one flow each side → one card, two buttons. Build runs `Ingest.cat`, Ask runs `DocChat.cat`, both immediately (`autoRun: true`). Details that follow from the same code:

- The button says **Build** when there's no index yet, **Rebuild** once `library.index/manifest.json` exists.
- **Ask is greyed out** until the index exists — nothing to read.
- If *two* flows built the same index, the Build button disappears and you get an orange note naming both. Same for Ask. The other side keeps working.
- The card prints "`Ingest.cat` builds · `DocChat.cat` queries" so you can see which file a click will run before you click it.

## 2. Not hardcoded — derived. But you do have to write the two rows

Two separate things here, and it's worth keeping them apart:

**Hardcoded:** the `ask_your_docs` workspace *itself* ships in the app. `BundledWorkspaces.swift` lists its five files (two `.cat` plus three sample PDFs) and copies them into `workspaces/ask_your_docs/` on first use. Deliberately **no prebuilt index** — Build is what creates it.

**Not hardcoded:** the Knowledge Base card. There is no `workspace.json`, no manifest, no naming convention, no "this is the ingest flow" setting. The comment in the file is explicit about it: classify each flow by what it does with an index, and when a builder and a querier name the same index, offer one card with two verbs.

So a user sets up their own by writing rows:

1. **Automate → AI Workflows → My Workspace → New Workspace**. You get `Workspace-xxxxxxxx/` with one starter `Flow.cat`.
2. Write one flow ending in `Store Index my.index`.
3. Write a second flow with `Read Index my.index` plus a `Retrieve` row.
4. The card appears by itself.

Names are matched loosely — `./library.index`, `library.index/` and `library.index` all pair up (leading `./` and trailing `/` are stripped). Or use **Import Workspace** to copy a folder in whole.

Two rules worth knowing because they'll bite:
- **One flow per side.** Two builders = no Build button, just a warning. That's on purpose.
- A card only shows if it has a builder **or** the index already exists on disk. A querier pointing at an index nothing builds and that doesn't exist yet shows nothing — the code calls it "a dead end, not a card." That's exactly why `uses_example` has a card with **Ask but no Build**: it ships a prebuilt `kb.index/` and nothing in it ever writes an index.

## 3. Yes — anything not a `.cat` is listed as a Shared File

The rule is literally: everything in the workspace folder, minus the flow files, minus hidden files (`.trash`, dotfiles). Directories included — that's why `docs/` and `kb.index/` show as folders with a folder icon.

But be careful about what "Shared File" means. **It's a listing, not a mechanism.** Nothing gets registered or activated by appearing there. Files are shared because of *where they sit*: every flow in a workspace resolves its relative paths against that one folder. `library.index` in `Ingest.cat` and `library.index` in `DocChat.cat` land on the same directory because both flows are scoped to the workspace, not to themselves. The Shared Files list is just showing you what's in the room.

Two limits: it only lists the **top level** (files nested inside `docs/` don't appear individually, though `docs/pattern=*.pdf` still reads them fine), and hidden folders are invisible — including `.trash`, which grows a full copy of the old index on every Rebuild.

## 4. Reveal in Finder and copy files in — yes, that works

That's the intended way, and there's no registration step. But "detected" splits into two different answers:

**For running flows: immediately.** No scan involved at all. A row like `Read Files docs/` resolves the path and reads the disk at run time. Drop a PDF into `docs/`, hit Build, it's in.

**For the UI list: on next look.** There's no file watcher on workspaces. `reloadWorkspaces()` runs at app launch, when the gallery appears, and when the workspace page appears. So navigating away and back refreshes it. A new `.cat` file you drop in needs that rescan before it shows in Flows (and before it counts toward a Build/Ask card); non-`.cat` shared files are read straight from disk each time the page draws.

Three things that will silently fail:

- **Copy real files, not aliases or symlinks.** `FlowWorkspace.resolve` refuses any symlink outright, even a working one — it's the security boundary that keeps a flow inside its folder. A Finder alias gets rejected the same way.
- **Absolute paths and `..` are refused** in `.cat` rows. Everything must be named relative to the workspace folder.
- **A folder with no `.cat` in it isn't a workspace** and the scan skips it entirely.

One practical note: use the **Reveal in Finder** button rather than typing the path. On the App Store build the app is sandboxed, so `~/Library/Application Support/AI Browser/workspaces/` actually resolves inside the app's container, not where you'd expect.
