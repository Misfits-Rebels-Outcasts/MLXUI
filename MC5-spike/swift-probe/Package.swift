// swift-tools-version: 6.3
import PackageDescription

let package = Package(
    name: "swift-probe",
    platforms: [.macOS(.v14)],
    dependencies: [
        // Pinned local checkout used by the MLXUI app (READ-ONLY).
        .package(
            path:
                "/Users/chinpinboo/Library/Developer/Xcode/DerivedData/MLXUI-agidnwckcxrlfyfzzahwlqamulby/SourcePackages/checkouts/mlx-swift"
        )
    ],
    targets: [
        .executableTarget(
            name: "probe",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/probe",
            swiftSettings: [.swiftLanguageMode(.v5)]
        )
    ]
)
