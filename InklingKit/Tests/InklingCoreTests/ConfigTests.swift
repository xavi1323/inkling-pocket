import Foundation
import Testing
@testable import InklingCore

let miniConfig = """
{"model_type": "inkling2", "eos_token_id": 200006,
 "text_config": {"model_type": "inkling2_loop_text", "hidden_size": 64, "num_hidden_layers": 8, "vocab_size": 512,
   "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "local_layer_ids": [0, 2, 3, 4, 5, 6],
   "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32, "n_layers_per_attn_res_block": 2,
   "loop_config": {"core_start": 2, "core_end": 6, "num_passes": 3, "gate_rank": 4, "is_varloop": false}},
 "quantization": {"group_size": 64, "bits": 4,
   "language_model.model.layers.2.mlp.switch_mlp.gate_proj": {"group_size": 32, "bits": 2, "mode": "affine"},
   "language_model.model.layers.2.self_attn.q_proj": {"group_size": 32, "bits": 8, "mode": "affine"},
   "language_model.model.layers.3.mlp.switch_mlp.up_proj": false}}
"""

@Test func decodesMiniConfig() throws {
    let config = try InklingModelConfig.decode(Data(miniConfig.utf8))
    #expect(config.text.hiddenSize == 64)
    #expect(config.text.layerVisits == [0, 1] + Array(2..<6) + Array(2..<6) + Array(2..<6) + [6, 7])
    #expect(config.text.isSliding(1) == false && config.text.isSliding(2))
    #expect(config.text.routeScale == 8.0)  // default from the Python dataclass
    #expect(config.eosTokenId == 200006)
    let q = config.quantization
    #expect(q.spec(for: "language_model.model.layers.2.mlp.switch_mlp.gate_proj") == QuantSpec(bits: 2, groupSize: 32, mode: "affine"))
    #expect(q.spec(for: "model.layers.2.self_attn.q_proj") == QuantSpec(bits: 8, groupSize: 32, mode: "affine"))
    #expect(q.spec(for: "language_model.model.layers.3.mlp.switch_mlp.up_proj") == nil)  // explicitly not quantized
    #expect(q.spec(for: "language_model.model.layers.5.self_attn.k_proj") == QuantSpec(bits: 4, groupSize: 64, mode: "affine"))
}

struct RealModel {
    static var folder: URL? { ProcessInfo.processInfo.environment["INKLING_MODEL_DIR"].map { URL(fileURLWithPath: $0) } }
    static func config() throws -> InklingModelConfig {
        try InklingModelConfig.decode(Data(contentsOf: try #require(folder).appending(path: "config.json")))
    }
}

@Suite(.enabled(if: RealModel.folder != nil, "set INKLING_MODEL_DIR to the model folder"))
struct RealConfigTests {
    @Test func layerVisitsHas48Entries() throws {
        let visits = try RealModel.config().text.layerVisits
        #expect(visits.count == 48)
        #expect(visits == Array(0..<6) + Array(6..<18) + Array(6..<18) + Array(6..<18) + Array(18..<24))
    }

    @Test func expertQuantIs2or3bit() throws {
        let q = try RealModel.config().quantization
        #expect(q.spec(for: "language_model.model.layers.6.mlp.switch_mlp.gate_proj") == QuantSpec(bits: 2, groupSize: 128, mode: "affine"))
        for layer in 0..<24 {
            let bits = try #require(q.spec(for: "language_model.model.layers.\(layer).mlp.switch_mlp.up_proj")).bits
            #expect(bits == 2 || bits == 3)
        }
    }

    @Test func attentionQuantIs8bit() throws {
        let q = try RealModel.config().quantization
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj", "r_proj", "gate_proj"] {
            #expect(q.spec(for: "language_model.model.layers.11.self_attn.\(proj)") == QuantSpec(bits: 8, groupSize: 64, mode: "affine"))
        }
        #expect(q.spec(for: "language_model.lm_head")?.bits == 6)
    }

    @Test func realTextConfigValues() throws {
        let t = try RealModel.config().text
        #expect(t.hiddenSize == 2304 && t.numExperts == 128 && t.expertsPerToken == 8 && t.intermediateSize == 1152)
        #expect(t.slidingWindowSize == 1024 && t.unpaddedVocabSize == 200058 && t.finalLogitSoftcapping == 30)
        #expect((0..<24).filter { !t.isSliding($0) } == [5, 11, 17, 23])
    }
}
