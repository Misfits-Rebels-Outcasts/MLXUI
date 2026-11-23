import AppKit
import PDFKit

/// ER-11 — the one `PrintPresenting` the app uses (SPEC-Q258, owner ruling R8): it opens the standard macOS print panel for a stored
/// outbox copy, and answers `completed` only when the person finished the print. It never prints silently (`showsPrintPanel` is
/// always on), and never runs `lp` or any other process. Printing from the App Store sandbox needs `com.apple.security.print` (R10).
///
/// `NSPrintOperation.run()` is `false` for both Cancel and a failed job, and AppKit does not say which, so a `false` is reported as
/// `.cancelled` (the entry stays pending with no error). Only a file the panel could not even be built for is `.failed`.
@MainActor
struct SystemPrintPresenter: PrintPresenting {
    func present(fileURL: URL, mime: String, title: String) async -> PrintPanelOutcome {
        let info = (NSPrintInfo.shared.copy() as? NSPrintInfo) ?? NSPrintInfo.shared
        let operation: NSPrintOperation
        if mime == "application/pdf" {
            guard let document = PDFDocument(url: fileURL), document.pageCount > 0,
                  let pdfOperation = document.printOperation(for: info, scalingMode: .pageScaleDownToFit, autoRotate: true) else {
                return .failed("the PDF could not be opened for printing")
            }
            operation = pdfOperation
        } else {
            guard let image = NSImage(contentsOf: fileURL), image.size.width > 0, image.size.height > 0 else {
                return .failed("the image could not be opened for printing")
            }
            // Fit the page's printable area (shrink only), centred.
            let usableWidth = info.paperSize.width - info.leftMargin - info.rightMargin
            let usableHeight = info.paperSize.height - info.topMargin - info.bottomMargin
            let scale = min(1, usableWidth / image.size.width, usableHeight / image.size.height)
            let view = NSImageView(frame: NSRect(x: 0, y: 0, width: image.size.width * scale, height: image.size.height * scale))
            view.image = image
            view.imageScaling = .scaleProportionallyUpOrDown
            info.isHorizontallyCentered = true
            info.isVerticallyCentered = true
            operation = NSPrintOperation(view: view, printInfo: info)
        }
        operation.jobTitle = title
        operation.showsPrintPanel = true
        operation.showsProgressPanel = true
        return operation.run() ? .completed : .cancelled
    }
}
