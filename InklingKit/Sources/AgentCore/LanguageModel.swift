import InklingCore

public struct GenerateParameters: Sendable, Equatable {
    public var temperature: Float = 0.7
    public var topP: Float = 0.95
    public var topK: Int = 0
    public var repetitionPenalty: Float = 1.0
    public var maxTokens: Int = 4096
    public init() {}
}

/// Anything that turns a chat into a stream of model events (real engine, echo model, test script).
public protocol LanguageModel: Sendable {
    func generate(_ messages: [ChatMessage], tools: [ToolSpec], effort: ReasoningEffort,
                  parameters: GenerateParameters) -> AsyncThrowingStream<ModelEvent, Error>
    func countTokens(_ text: String) -> Int
}
