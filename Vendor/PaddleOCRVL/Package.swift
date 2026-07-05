// swift-tools-version:5.9
import PackageDescription

// Vendored from https://github.com/mlx-community/paddleocr-vl.swift (MIT, © 2025 lulzx).
// Pins bumped to match MLXUI's resolved graph (mlx-swift 0.31.x /
// swift-transformers 1.x); the upstream 0.29.1 / 0.1.21 pins conflict with the host app.
// CLI target + swift-argument-parser dropped — the app only needs the library.
let package = Package(
    name: "PaddleOCRVL",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "PaddleOCRVL", targets: ["PaddleOCRVL"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.4")),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "PaddleOCRVL",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers")
            ]
        )
    ]
)
