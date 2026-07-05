import SwiftUI

struct SourceBadge: View {
    let source: ModelSource

    var body: some View {
        Text(source.rawValue.uppercased())
            .font(.caption2)
            .fontWeight(.bold)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .foregroundStyle(color)
            .clipShape(RoundedRectangle(cornerRadius: 4))
    }

    private var color: Color {
        switch source {
        case .mlx: .blue
        case .coreai: .purple
        case .coreml: .green
        case .research: .orange
        }
    }
}

struct StarRating: View {
    let rating: Double

    var body: some View {
        HStack(spacing: 1) {
            ForEach(1...5, id: \.self) { i in
                Image(systemName: Double(i) <= rating + 0.25 ? "star.fill" :
                      Double(i) <= rating + 0.75 ? "star.leadinghalf.filled" : "star")
                    .font(.caption2)
                    .foregroundStyle(.yellow)
            }
            Text(String(format: "%.1f", rating))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}
