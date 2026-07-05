# PaddleOCR-VL.swift

A native Swift implementation of [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) using [MLX](https://github.com/ml-explore/mlx-swift) for Apple Silicon.

PaddleOCR-VL is a 0.9B ultra-compact vision-language model optimized for document understanding, supporting text recognition, table parsing, formula extraction, and chart analysis across 109 languages.

## Features

- Native Swift implementation optimized for Apple Silicon
- Automatic model downloading from HuggingFace Hub
- Support for multiple document understanding tasks
- NaViT-style dynamic resolution processing
- Batch processing for multiple images
- Both library and CLI interfaces

## Requirements

- macOS 14.0+
- Apple Silicon (M1/M2/M3/M4)
- Swift 5.9+

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/lulzx/paddleocr-vl.swift", from: "1.0.0")
]
```

Then add `PaddleOCRVL` to your target:

```swift
.target(
    name: "YourTarget",
    dependencies: ["PaddleOCRVL"]
)
```

### Building from Source

```bash
git clone https://github.com/lulzx/paddleocr-vl.swift
cd paddleocr-vl.swift
swift build -c release
```

The CLI binary will be at `.build/release/PaddleOCRVLCLI`.

## Quick Start

### Command Line

```bash
# Basic OCR - model downloads automatically on first run
paddleocr-vl ocr document.png

# Table recognition
paddleocr-vl ocr spreadsheet.png --task table

# Formula recognition
paddleocr-vl ocr equation.png --task formula

# Chart data extraction
paddleocr-vl ocr chart.png --task chart

# Batch processing with verbose output
paddleocr-vl ocr *.png --task ocr --verbose

# Save output to file
paddleocr-vl ocr document.png --output result.txt
```

### Swift API

```swift
import PaddleOCRVL

// Initialize pipeline (downloads model if needed)
let pipeline = try await PaddleOCRVLPipeline(modelPath: "PaddlePaddle/PaddleOCR-VL")

// Recognize text from an image
let text = try pipeline.recognize(imagePath: "document.png")
print(text)

// Table recognition
let tableMarkdown = try pipeline.recognize(imagePath: "table.png", task: .table)

// Batch processing
let results = try pipeline.recognizeBatch(imagePaths: ["page1.png", "page2.png"])

// Process CIImage directly
let ciImage = CIImage(contentsOf: imageURL)!
let result = pipeline.recognize(image: ciImage, task: .formula)
```

## Supported Tasks

| Task | Description | Output Format |
|------|-------------|---------------|
| `ocr` | General text recognition | Plain text |
| `table` | Table structure recognition | Markdown/HTML table |
| `formula` | Mathematical formula recognition | LaTeX |
| `chart` | Chart data extraction | Structured data |

## CLI Options

```
USAGE: paddleocr-vl ocr <image-paths> ... [--model <model>] [--output <output>] [--task <task>] [--max-tokens <max-tokens>] [--mode <mode>] [--verbose] [--cache-limit <cache-limit>]

ARGUMENTS:
  <image-paths>           Path(s) to input image file(s)

OPTIONS:
  -m, --model <model>     Model path or HuggingFace ID (default: PaddlePaddle/PaddleOCR-VL)
  -o, --output <output>   Path to save output text
  -t, --task <task>       Task type: ocr, table, formula, chart (default: ocr)
  --max-tokens <n>        Maximum tokens to generate (default: 1024)
  --mode <mode>           Processing mode: base (448x448), dynamic (NaViT) (default: base)
  --verbose               Show timing and progress information
  --cache-limit <mb>      GPU memory cache limit in MB
  -h, --help              Show help information
```

## Processing Modes

- **base**: Fixed 448x448 resolution. Faster, suitable for most documents.
- **dynamic**: NaViT-style adaptive resolution. Better for high-resolution images with fine details.

## Model Specification

The model can be specified as:
- **HuggingFace ID**: `PaddlePaddle/PaddleOCR-VL` (auto-downloads)
- **HuggingFace ID with revision**: `PaddlePaddle/PaddleOCR-VL:main`
- **Local path**: `/path/to/model` or `./models/paddleocr-vl`

Models are cached in `~/.cache/huggingface/hub/` and reused automatically.

## Supported Image Formats

JPEG, PNG, GIF, BMP, TIFF, WebP

## Architecture

```
PaddleOCRVL/
├── VisionEncoder      # NaViT-style dynamic resolution encoder
├── MultiModalProjector # Vision-to-language projection
├── LanguageModel      # ERNIE-4.5-0.3B decoder
├── ImageProcessor     # Image preprocessing pipeline
├── Generator          # Token generation with sampling
└── Pipeline           # High-level API
```

## Dependencies

- [mlx-swift](https://github.com/ml-explore/mlx-swift) - Apple's ML framework for Apple Silicon
- [swift-transformers](https://github.com/huggingface/swift-transformers) - Tokenizer support
- [swift-argument-parser](https://github.com/apple/swift-argument-parser) - CLI argument parsing

## License

MIT License
