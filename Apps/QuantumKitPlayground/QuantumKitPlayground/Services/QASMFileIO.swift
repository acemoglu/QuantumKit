import Foundation
import SwiftUI
import UniformTypeIdentifiers

enum PlaygroundFileError: LocalizedError {
    case tooLarge
    case unreadable
    case unsupportedExtension(String)

    var errorDescription: String? {
        switch self {
        case .tooLarge:
            return "File is larger than 1 MB and cannot be opened."
        case .unreadable:
            return "Could not read the file as UTF-8 text."
        case .unsupportedExtension(let ext):
            return "Unsupported file type “\(ext)”. Drop or open a .qasm or .txt file."
        }
    }
}

enum QASMFileIO {
    static let maxFileBytes = 1_048_576
    static let allowedExtensions: Set<String> = ["qasm", "txt"]

    static var qasmContentType: UTType {
        UTType(filenameExtension: "qasm") ?? .plainText
    }

    static var readableContentTypes: [UTType] {
        var types: [UTType] = [qasmContentType, .plainText, .utf8PlainText, .text]
        return types
    }

    static func load(from url: URL) throws -> String {
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let ext = url.pathExtension.lowercased()
        if !ext.isEmpty {
            guard allowedExtensions.contains(ext) else {
                throw PlaygroundFileError.unsupportedExtension(ext)
            }
        }

        let values = try url.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
        if let isRegular = values.isRegularFile, !isRegular {
            throw PlaygroundFileError.unreadable
        }
        if let size = values.fileSize, size > maxFileBytes {
            throw PlaygroundFileError.tooLarge
        }

        let data = try Data(contentsOf: url)
        if data.count > maxFileBytes {
            throw PlaygroundFileError.tooLarge
        }
        guard let text = String(data: data, encoding: .utf8) else {
            throw PlaygroundFileError.unreadable
        }
        return text
    }

    static func save(_ text: String, to url: URL) throws {
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    static func canConsume(_ providers: [NSItemProvider]) -> Bool {
        providers.contains { $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) }
    }

    static func loadDropped(from providers: [NSItemProvider]) async throws -> String {
        guard let provider = providers.first(where: {
            $0.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier)
        }) else {
            throw PlaygroundFileError.unreadable
        }

        let url: URL = try await withCheckedThrowingContinuation { continuation in
            provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier, options: nil) { item, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                guard let url = Self.url(fromDropItem: item) else {
                    continuation.resume(throwing: PlaygroundFileError.unreadable)
                    return
                }
                continuation.resume(returning: url)
            }
        }
        return try load(from: url)
    }

    private static func url(fromDropItem item: NSSecureCoding?) -> URL? {
        if let url = item as? URL {
            return url
        }
        if let url = item as? NSURL {
            return url as URL
        }
        if let data = item as? Data {
            return URL(dataRepresentation: data, relativeTo: nil)
        }
        return nil
    }
}

struct QASMFileDocument: FileDocument {
    static var readableContentTypes: [UTType] { QASMFileIO.readableContentTypes }
    static var writableContentTypes: [UTType] { [QASMFileIO.qasmContentType, .plainText] }

    var text: String

    init(text: String = "") {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        guard let data = configuration.file.regularFileContents else {
            throw PlaygroundFileError.unreadable
        }
        if data.count > QASMFileIO.maxFileBytes {
            throw PlaygroundFileError.tooLarge
        }
        guard let string = String(data: data, encoding: .utf8) else {
            throw PlaygroundFileError.unreadable
        }
        self.text = string
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}
