// swift-tools-version: 6.0
import PackageDescription

// Pure-logic targets build everywhere (Linux tests on the dev PC and CI).
var products: [Product] = [
    .library(name: "InklingCore", targets: ["InklingCore"]),
    .library(name: "AgentCore", targets: ["AgentCore"]),
    .library(name: "CrawlCore", targets: ["CrawlCore"]),
    .library(name: "OpenUICore", targets: ["OpenUICore"]),
]
var dependencies: [Package.Dependency] = [
    .package(url: "https://github.com/scinfu/SwiftSoup.git", exact: "2.14.4"),
]
var targets: [Target] = [
    .target(name: "InklingCore"),
    .target(name: "AgentCore", dependencies: ["InklingCore"]),
    .target(name: "CrawlCore", dependencies: ["AgentCore", "SwiftSoup"]),
    .target(name: "OpenUICore"),
    .testTarget(name: "InklingCoreTests", dependencies: ["InklingCore"], resources: [.copy("Fixtures")]),
    .testTarget(name: "AgentCoreTests", dependencies: ["AgentCore"]),
    .testTarget(name: "CrawlCoreTests", dependencies: ["CrawlCore"], resources: [.copy("Fixtures")]),
    .testTarget(name: "OpenUICoreTests", dependencies: ["OpenUICore"], resources: [.copy("Fixtures")]),
]

#if !os(Linux)
// Apple-only targets: MLX engine, WebKit, SwiftData, SwiftUI.
products += [
    .library(name: "InklingEngine", targets: ["InklingEngine"]),
    .library(name: "PocketCrawler", targets: ["PocketCrawler"]),
    .library(name: "PocketMemory", targets: ["PocketMemory"]),
    .library(name: "OpenUIViews", targets: ["OpenUIViews"]),
]
dependencies += [
    .package(url: "https://github.com/ml-explore/mlx-swift.git", exact: "0.33.0"),
]
targets += [
    .target(name: "InklingEngine", dependencies: [
        "InklingCore", "AgentCore",
        .product(name: "MLX", package: "mlx-swift"),
        .product(name: "MLXNN", package: "mlx-swift"),
        .product(name: "MLXFast", package: "mlx-swift"),
    ]),
    .target(name: "PocketCrawler", dependencies: ["CrawlCore"]),
    .target(name: "PocketMemory", dependencies: ["AgentCore", "InklingCore"]),
    .target(name: "OpenUIViews", dependencies: ["OpenUICore"]),
    .testTarget(name: "InklingEngineTests", dependencies: ["InklingEngine"], resources: [.copy("Fixtures")]),
    .testTarget(name: "PocketCrawlerTests", dependencies: ["PocketCrawler"]),
    .testTarget(name: "PocketMemoryTests", dependencies: ["PocketMemory", "InklingCore"]),
    .testTarget(name: "OpenUIViewsTests", dependencies: ["OpenUIViews"]),
]
#endif

let package = Package(
    name: "InklingKit",
    platforms: [.iOS("26.0"), .macOS("26.0")],
    products: products,
    dependencies: dependencies,
    targets: targets
)
