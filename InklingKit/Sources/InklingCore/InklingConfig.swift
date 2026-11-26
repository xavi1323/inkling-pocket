// Model configuration for inkling2_loop, mirroring mlx-vlm inkling2/config.py (MIT, Prince Canuma) at d998a62.
import Foundation

public struct LoopConfig: Decodable, Sendable, Equatable {
    public var coreStart: Int
    public var coreEnd: Int
    public var numPasses: Int
    public var gateRank: Int
    public var isVarloop: Bool

    enum CodingKeys: String, CodingKey { case coreStart = "core_start", coreEnd = "core_end", numPasses = "num_passes", gateRank = "gate_rank", isVarloop = "is_varloop" }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        coreStart = try c.decode(Int.self, forKey: .coreStart)
        coreEnd = try c.decode(Int.self, forKey: .coreEnd)
        numPasses = try c.decode(Int.self, forKey: .numPasses)
        gateRank = try c.decode(Int.self, forKey: .gateRank)
        isVarloop = try c.decodeIfPresent(Bool.self, forKey: .isVarloop) ?? false
    }
}

/// `text_config` with the Python dataclass defaults for missing keys.
public struct InklingTextConfig: Decodable, Sendable {
    public var modelType: String
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var vocabSize: Int
    public var unpaddedVocabSize: Int?
    public var rmsNormEps: Float
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var headDim: Int
    public var swaNumAttentionHeads: Int
    public var swaNumKeyValueHeads: Int
    public var swaHeadDim: Int
    public var slidingWindowSize: Int
    public var localLayerIds: [Int]?
    public var layerTypes: [String]?
    public var dRel: Int
    public var relExtent: Int
    public var logScalingNFloor: Int?
    public var logScalingAlpha: Float
    public var sconvKernelSize: Int
    public var intermediateSize: Int
    public var numExperts: Int
    public var expertsPerToken: Int
    public var numSharedExperts: Int
    public var routeScale: Float
    public var qkL2ClipThreshold: Float
    public var relL2ClipThreshold: Float
    public var layersPerAttnResBlock: Int
    public var attnResNormalizeValues: Bool
    public var useValueNormNoGamma: Bool
    public var mlpGateSoftcap: Float
    public var mlpUpSoftcap: Float
    public var finalLogitSoftcapping: Float
    public var useGateBias: Bool
    public var loopConfig: LoopConfig?

    enum K: String, CodingKey {
        case model_type, hidden_size, num_hidden_layers, vocab_size, unpadded_vocab_size, rms_norm_eps
        case num_attention_heads, num_key_value_heads, head_dim, swa_num_attention_heads, swa_num_key_value_heads, swa_head_dim
        case sliding_window_size, local_layer_ids, layer_types, d_rel, rel_extent, log_scaling_n_floor, log_scaling_alpha
        case sconv_kernel_size, intermediate_size, n_routed_experts, num_experts_per_tok, n_shared_experts, route_scale
        case qk_l2_clip_threshold, rel_l2_clip_threshold, n_layers_per_attn_res_block, attn_res_normalize_values
        case use_value_norm_no_gamma, mlp_gate_softcap, mlp_up_softcap, final_logit_softcapping, use_gate_bias, loop_config
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: K.self)
        func v<T: Decodable>(_ k: K, _ d: T) throws -> T { try c.decodeIfPresent(T.self, forKey: k) ?? d }
        modelType = try v(.model_type, "inkling2_text")
        hiddenSize = try v(.hidden_size, 4608)
        numHiddenLayers = try v(.num_hidden_layers, 48)
        vocabSize = try v(.vocab_size, 201_024)
        unpaddedVocabSize = c.contains(.unpadded_vocab_size) ? try c.decodeIfPresent(Int.self, forKey: .unpadded_vocab_size) : 200_058
        rmsNormEps = try v(.rms_norm_eps, 1e-6)
        numAttentionHeads = try v(.num_attention_heads, 48)
        numKeyValueHeads = try v(.num_key_value_heads, 8)
        headDim = try v(.head_dim, 128)
        swaNumAttentionHeads = try v(.swa_num_attention_heads, numAttentionHeads)
        swaNumKeyValueHeads = try v(.swa_num_key_value_heads, numKeyValueHeads)
        swaHeadDim = try v(.swa_head_dim, headDim)
        slidingWindowSize = try v(.sliding_window_size, 1024)
        localLayerIds = try c.decodeIfPresent([Int].self, forKey: .local_layer_ids)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layer_types)
        dRel = try v(.d_rel, 16)
        relExtent = try v(.rel_extent, 1024)
        logScalingNFloor = c.contains(.log_scaling_n_floor) ? try c.decodeIfPresent(Int.self, forKey: .log_scaling_n_floor) : 128_000
        logScalingAlpha = try v(.log_scaling_alpha, 0.1)
        sconvKernelSize = try v(.sconv_kernel_size, 4)
        intermediateSize = try v(.intermediate_size, 2304)
        numExperts = try v(.n_routed_experts, 384)
        expertsPerToken = try v(.num_experts_per_tok, 8)
        numSharedExperts = try v(.n_shared_experts, 0)
        routeScale = try v(.route_scale, 8.0)
        qkL2ClipThreshold = try v(.qk_l2_clip_threshold, 22.627416997969522)
        relL2ClipThreshold = try v(.rel_l2_clip_threshold, 8.0)
        layersPerAttnResBlock = try v(.n_layers_per_attn_res_block, 6)
        attnResNormalizeValues = try v(.attn_res_normalize_values, true)
        useValueNormNoGamma = try v(.use_value_norm_no_gamma, true)
        mlpGateSoftcap = try v(.mlp_gate_softcap, 4.0)
        mlpUpSoftcap = try v(.mlp_up_softcap, 25.0)
        finalLogitSoftcapping = try v(.final_logit_softcapping, 30.0)
        useGateBias = try v(.use_gate_bias, true)
        loopConfig = try c.decodeIfPresent(LoopConfig.self, forKey: .loop_config)
    }

    /// Layer index for each of the model's visits: prefix, core × passes, suffix (48 for Inkling 2 Mini).
    public var layerVisits: [Int] {
        let all = Array(0..<numHiddenLayers)
        guard let loop = loopConfig else { return all }
        return Array(all[..<loop.coreStart])
            + Array(repeating: Array(all[loop.coreStart..<loop.coreEnd]), count: loop.numPasses).flatMap { $0 }
            + Array(all[loop.coreEnd...])
    }

    public func isSliding(_ layer: Int) -> Bool {
        if let types = layerTypes { return types[layer] == "hybrid_sliding" }
        if let ids = localLayerIds { return ids.contains(layer) }
        return (layer + 1) % 6 != 0
    }

    /// Vocabulary actually produced (logits past this are padding).
    public var outputVocabSize: Int { unpaddedVocabSize ?? vocabSize }
}

public struct QuantSpec: Sendable, Equatable {
    public var bits: Int
    public var groupSize: Int
    public var mode: String
    public init(bits: Int, groupSize: Int, mode: String) { self.bits = bits; self.groupSize = groupSize; self.mode = mode }
}

/// Per-module quantization from config.json `quantization` (top-level default + per-path overrides; `false` = not quantized).
public struct QuantizationMap: Sendable {
    public var defaultSpec: QuantSpec?
    var overrides: [String: QuantSpec?] = [:]

    public init(_ json: JSONValue?) {
        guard case .object(let entries)? = json else { return }
        var bits: Int?, group: Int?, mode = "affine"
        for (key, value) in entries {
            switch (key, value) {
            case ("bits", .int(let b)): bits = Int(b)
            case ("group_size", .int(let g)): group = Int(g)
            case ("mode", .string(let m)): mode = m
            case (_, .bool(false)): overrides[key] = .some(nil)
            case (_, .object):
                if case .int(let b)? = value["bits"], case .int(let g)? = value["group_size"] {
                    overrides[key] = QuantSpec(bits: Int(b), groupSize: Int(g), mode: value["mode"]?.stringValue ?? "affine")
                }
            default: break
            }
        }
        if let bits, let group { defaultSpec = QuantSpec(bits: bits, groupSize: group, mode: mode) }
    }

    /// Spec for a module path (with or without the `language_model.` prefix); the default when not listed.
    public func spec(for modulePath: String) -> QuantSpec? {
        for candidate in [modulePath, "language_model." + modulePath] {
            if let entry = overrides[candidate] { return entry }
        }
        return defaultSpec
    }
}

public struct InklingModelConfig: Sendable {
    public var modelType: String
    public var text: InklingTextConfig
    public var quantization: QuantizationMap
    public var eosTokenId: Int

    public static func decode(_ data: Data) throws -> InklingModelConfig {
        struct Top: Decodable {
            var model_type: String?
            var text_config: InklingTextConfig?
            var eos_token_id: Int?
        }
        let top = try JSONDecoder().decode(Top.self, from: data)
        let json = try JSONValue.parse(String(decoding: data, as: UTF8.self))
        let text = try top.text_config ?? JSONDecoder().decode(InklingTextConfig.self, from: Data("{}".utf8))
        return InklingModelConfig(modelType: top.model_type ?? "inkling2", text: text,
                                  quantization: QuantizationMap(json["quantization"] ?? json["quantization_config"]),
                                  eosTokenId: top.eos_token_id ?? 200_006)
    }
}
