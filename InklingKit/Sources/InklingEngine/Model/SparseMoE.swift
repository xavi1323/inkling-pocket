// Ported from mlx-vlm inkling2/language.py SparseMoE + switch_layers.SwitchGLU (MIT) at d998a62.
import MLX
import InklingCore

/// Sigmoid-routed top-k experts with a softcapped SwiGLU; route weights are renormalised and scaled.
final class SparseMoE {
    let gateWeight: MLXArray  // [E, D]
    /// `gateWeight` in FP32, converted once at load (not on every visit; slotstream measured ~2% for this).
    let gateFP32: MLXArray
    let scoreBias: MLXArray   // [E]
    let gateProj, upProj, downProj: QSwitchLinear
    /// Streaming slots; when set the projections above read the slots, indexed by slot instead of expert.
    let cache: ExpertCache?
    let topK: Int
    let routeScale, gateCap, upCap: Float
    /// Single-token kernels over the same weights (fast path).
    lazy var fused = FusedMoEDecode(gate: gateProj.linear, up: upProj.linear, down: downProj.linear, gateCap: gateCap, upCap: upCap)

    init(_ config: InklingTextConfig, layer: Int, path: String, loader: WeightLoader) throws {
        gateWeight = try loader.tensor(path + ".gate_weight")
        scoreBias = try loader.tensor(path + ".e_score_correction_bias")
        gateFP32 = gateWeight.asType(.float32)
        eval(gateFP32)
        if let plan = loader.expertSlots, layer < plan.count, let mapped = loader.source as? MappedSafetensors {
            let slots = plan[layer]
            let cache = try ExpertCache(path: path, weights: mapped, quant: loader.quant, prefix: loader.prefix,
                                        slots: min(max(slots, config.expertsPerToken), config.numExperts))
            self.cache = cache
            (gateProj, upProj, downProj) = (QSwitchLinear(linear: cache.gate), QSwitchLinear(linear: cache.up), QSwitchLinear(linear: cache.down))
        } else {
            cache = nil
            gateProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.gate_proj"))
            upProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.up_proj"))
            downProj = QSwitchLinear(linear: try loader.linear(path + ".switch_mlp.down_proj"))
        }
        topK = config.expertsPerToken
        routeScale = config.routeScale
        gateCap = config.mlpGateSoftcap
        upCap = config.mlpUpSoftcap
    }

    /// Expert ids [B,L,k] and their weights [B,L,k] (FP32).
    func route(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let scores = sigmoid(matmul(x.asType(.float32), gateFP32.T))
        let indices = argPartition(-(scores + scoreBias), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
        return (indices, weights)
    }

    /// Smart expert picks (cache-aware routing with the top experts guaranteed; Skliar et al., arXiv 2412.00099):
    /// the router's best `keepTop` experts always run; the other picks come from its best `pickFrom`, those in a slot
    /// (`resident` [E], 1 = in a slot) first, then the rest by rank. Weights are the router's for what runs,
    /// renormalised as usual. Swapping only lower-ranked experts barely changes a fine-grained MoE's output, and the
    /// cache keeps the experts it reuses, so far fewer are read from storage.
    func smartRoute(_ x: MLXArray, resident: MLXArray, keepTop: Int, pickFrom: Int) -> (MLXArray, MLXArray) {
        let scores = sigmoid(matmul(x.asType(.float32), gateFP32.T))
        let order = argSort(-(scores + scoreBias), axis: -1)
        let rank = argSort(order, axis: -1).asType(.float32)  // each expert's rank, 0 = best
        let pool = Float(pickFrom)
        let key = which(rank .< Float(keepTop), rank - 1e6, which(rank .< pool, rank + (1 - resident) * pool, rank + 1e6))
        let indices = argPartition(key, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
        return (indices, weights)
    }

    /// `smartRoute` compiled (a few fused kernels instead of a dozen); one per `keepTop`. GPU only.
    private var compiledSmart: (keepTop: Int, fn: ([MLXArray]) -> [MLXArray])?

    /// The smart router for this step: compiled on the fast path.
    func smartRoute(_ x: MLXArray, resident: MLXArray, fast: Bool) -> (MLXArray, MLXArray) {
        guard fast, Ops.useFused else { return smartRoute(x, resident: resident, keepTop: smartKeepTop, pickFrom: Self.smartPickFrom) }
        if compiledSmart?.keepTop != smartKeepTop {
            let keep = smartKeepTop
            compiledSmart = (keep, compile(inputs: [], outputs: []) { [unowned self] a in
                let (i, w) = smartRoute(a[0], resident: a[1], keepTop: keep, pickFrom: Self.smartPickFrom)
                return [i, w]
            })
        }
        let r = compiledSmart!.fn([x, resident])
        return (r[0], r[1])
    }

    /// Smart picks on (decode steps and speculative checks only; prompt chunks stay exact).
    var smartRouting = false
    static let smartKeepTopDefault = 3
    var smartKeepTop = smartKeepTopDefault
    static let smartPickFrom = 16
    static let smartMaxTokens = 16

    /// `route` compiled (one token's router as a few fused kernels); GPU only.
    private lazy var compiledRoute: (MLXArray) -> (MLXArray, MLXArray) = compile(inputs: [], outputs: []) { [unowned self] x in
        route(x)
    }

    /// This router's top-k experts for `x`, best first (a guess at a later visit's routing).
    func rankedGuess(_ x: MLXArray) -> MLXArray {
        let scores = sigmoid(matmul(x.asType(.float32), gateFP32.T)) + scoreBias
        return argSort(-scores, axis: -1)[.ellipsis, ..<topK]
    }

    private lazy var compiledGuess: (MLXArray) -> MLXArray = compile(inputs: [], outputs: []) { [unowned self] x in
        rankedGuess(x)
    }

    /// Turbo: this token's expert table (`ExpertCache.turboTable` as arrays), set between tokens; nil = exact.
    var turboTable: (slots: MLXArray, ready: MLXArray)?
    /// Turbo: (wanted, used) experts of each visit this token, read back between tokens.
    var turboRoutes: [(wanted: MLXArray, used: MLXArray)] = []
    /// Turbo: share of each visit's routing weight that went to swapped experts this token ([1] each).
    var turboMisses: [MLXArray] = []

    /// Turbo router: [wanted ids, used ids, their slots, weights, missed share]. Experts not ready are replaced by the
    /// best ready ones; weights are renormalised over what runs (a missing expert gets weight 0). Missed share: the
    /// part of the wanted experts' routing weight that was not ready.
    func turboRoute(_ a: [MLXArray]) -> [MLXArray] {
        let (x, slots, ready) = (a[0], a[1], a[2])
        let scores = sigmoid(matmul(x.asType(.float32), gateFP32.T))
        let selection = scores + scoreBias
        let wanted = argPartition(-selection, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let used = argPartition(-(selection + (ready - 1) * 1e9), kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores * ready, used, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20) * routeScale
        let want = takeAlong(scores, wanted, axis: -1)
        let missed = (want * (1 - take(ready, wanted))).sum() / (want.sum() + 1e-20)
        return [wanted, used, take(slots, used), weights, missed.reshaped([1])]
    }

    private lazy var compiledTurbo: ([MLXArray]) -> [MLXArray] = compile(inputs: [], outputs: []) { [unowned self] a in
        turboRoute(a)
    }

    /// The router for this step: compiled for single tokens on the fast path.
    func route(_ x: MLXArray, fast: Bool) -> (MLXArray, MLXArray) {
        fast && Ops.useFused && x.dim(0) * x.dim(1) == 1 ? compiledRoute(x) : route(x)
    }

    func activation(up: MLXArray, gate: MLXArray) -> MLXArray {
        let g = gate.asType(.float32), u = up.asType(.float32)
        return (tanh(g / gateCap) * gateCap * sigmoid(g) * tanh(u / upCap) * upCap).asType(gate.dtype)
    }

    /// The MoE of the next layer visit (set by TextModel before each visit): its experts are read ahead.
    weak var next: SparseMoE?
    /// Benchmark only: single-token steps use the first `topK` slots without reading the routing back (wrong
    /// answers; shows the speed with no routing waits or expert reads).
    var ignoreRouting = false

    func callAsFunction(_ x: MLXArray, onRoute: ((MLXArray) -> Void)? = nil, fast: Bool = false) -> MLXArray {
        if let table = turboTable, cache != nil, !ignoreRouting, x.dim(0) * x.dim(1) == 1 {
            let inputs = [x, table.slots, table.ready]
            let r = fast && Ops.useFused ? compiledTurbo(inputs) : turboRoute(inputs)
            onRoute?(r[1])
            turboRoutes.append((r[0], r[1]))
            turboMisses.append(r[4])
            return experts(x, r[2], r[3], fast: fast)
        }
        let smart = smartRouting && cache != nil && !ignoreRouting && x.dim(0) * x.dim(1) <= Self.smartMaxTokens
        let (indices, weights) = smart ? smartRoute(x, resident: cache!.residentMask(), fast: fast) : route(x, fast: fast)
        onRoute?(indices)
        guard let cache else { return experts(x, indices, weights, fast: fast) }
        let (b, length, d) = (x.dim(0), x.dim(1), x.dim(2))
        if ignoreRouting, b * length == 1 {
            let out = experts(x, MLXArray(Array(0..<Int32(topK))).reshaped(indices.shape), weights, fast: fast)
            asyncEval(out)
            return out
        }
        // Single tokens and speculative checks: guess the next visit's experts from this input (its router on our
        // input) and read them in while the GPU works. Safe: everything queued before reading the routing back has
        // finished, and the work queued after it only reads this layer's slots.
        let rows = b * length
        let guess = rows <= Self.smartMaxTokens && next?.cache != nil
            ? (fast && Ops.useFused ? next!.compiledGuess(x) : next!.rankedGuess(x)) : nil
        // Prompt chunks: the next visit's router on this chunk, to read its experts in while this visit computes.
        let chunkGuess = rows > Self.smartMaxTokens && next?.cache != nil && next?.cache !== cache ? next!.rankedGuess(x) : nil
        let waited = [indices] + (guess.map { [$0] } ?? []) + (chunkGuess.map { [$0] } ?? [])
        if let profile = cache.profile { profile.time(\.gpuWait) { eval(waited) } } else { eval(waited) }
        let ids = indices.asArray(Int32.self)
        if let guess, let nextCache = next?.cache {
            let (experts, ranks) = Self.rankMajor(guess.asArray(Int32.self).map(Int.init), rows: rows, k: topK,
                                                  depth: rows == 1 ? topK : Self.checkGuessDepth)
            nextCache.prefetch(experts, ranks: ranks)
        }
        var unique: [Int] = []
        var seen = Set<Int>()
        for id in ids where seen.insert(Int(id)).inserted { unique.append(Int(id)) }
        // Once this visit's work is queued: read the next visit's guessed experts while the GPU runs it.
        func readAheadNext() {
            guard let chunkGuess, let nextCache = next?.cache else { return }
            nextCache.readAhead(Self.byCount(chunkGuess.asArray(Int32.self)))
        }
        if unique.count <= cache.slots {
            let map = cache.ensure(unique)
            let out = experts(x, MLXArray(ids.map { map[Int($0)]! }).reshaped(indices.shape), weights, fast: fast)
            asyncEval(out)  // start the GPU while the next layer's graph is built
            readAheadNext()
            return out
        }
        // More experts than slots (long prompt chunks): run them in groups, each over only the routes it serves.
        // The experts already in a slot go first; then the missing ones in batches of at most half the slots, each
        // read into slots the batch still on the GPU does not use, so reading overlaps the GPU.
        let k = indices.dim(-1)
        let flatX = x.reshaped([b * length, d]), flatW = weights.reshaped([-1])
        var total = MLXArray.zeros([b * length, d], dtype: .float32)
        let resident = cache.resident(unique)
        var missing = unique.filter { !resident.contains($0) }
        var inFlight: [(slots: Set<Int>, done: MLXArray)] = []  // queued batches, oldest first
        func run(_ group: [Int]) {
            let busy = inFlight.reduce(into: Set<Int>()) { $0.formUnion($1.slots) }
            let map = cache.ensure(group, avoiding: busy)
            var routes: [(position: Int32, slot: Int32)] = []
            for (i, id) in ids.enumerated() { if let s = map[Int(id)] { routes.append((Int32(i), s)) } }
            routes.sort { $0.slot < $1.slot }
            let tokens = MLXArray(routes.map { $0.position / Int32(k) })
            let slots = MLXArray(routes.map(\.slot))
            let rows = take(flatX, tokens, axis: 0).expandedDimensions(axis: 1)
            let up = upProj(rows, slots, sorted: true)
            let gate = gateProj(rows, slots, sorted: true)
            let down = downProj(activation(up: up, gate: gate), slots, sorted: true).reshaped([routes.count, d])
            let w = take(flatW, MLXArray(routes.map(\.position)), axis: 0).expandedDimensions(axis: -1)
            total = total.at[tokens].add(down.asType(.float32) * w)
            asyncEval(total)
            inFlight.append((Set(map.values.map(Int.init)), total))
        }
        let firstGroup = unique.filter { resident.contains($0) }
        if !firstGroup.isEmpty { run(firstGroup) }
        let batch = max(cache.slots / 2, 1)
        while !missing.isEmpty {
            // Free enough slots: wait for the oldest batches (the GPU keeps running the newer one meanwhile).
            while let oldest = inFlight.first,
                  cache.slots - inFlight.reduce(into: Set<Int>(), { $0.formUnion($1.slots) }).count < min(batch, missing.count) {
                eval(oldest.done)
                inFlight.removeFirst()
            }
            let busy = inFlight.reduce(into: Set<Int>()) { $0.formUnion($1.slots) }.count
            let group = Array(missing.prefix(min(batch, cache.slots - busy)))
            missing.removeFirst(group.count)
            run(group)
        }
        // The GPU still runs the last batches and storage is idle. The next visit's slots are free: its earlier
        // visits finished before our routing was read back.
        readAheadNext()
        return total.reshaped([b, length, d]).asType(x.dtype)
    }

    /// Distinct experts in `guess`, most often guessed first (ties: lower id).
    static func byCount(_ guess: [Int32]) -> [Int] {
        var counts: [Int: Int] = [:]
        for e in guess { counts[Int(e), default: 0] += 1 }
        return counts.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }.map(\.key)
    }

    /// How many of each token's best guesses a multi-token check reads ahead (smart picks always run the top 3).
    static let checkGuessDepth = 4

    /// Row-major guesses [rows × k] (best first per row) → distinct experts ordered by rank across rows, each with the
    /// best rank any row gave it; ranks below `depth` only.
    static func rankMajor(_ guess: [Int], rows: Int, k: Int, depth: Int) -> ([Int], [Int]) {
        var experts: [Int] = [], ranks: [Int] = [], seen = Set<Int>()
        for r in 0..<min(depth, k) {
            for row in 0..<rows where seen.insert(guess[row * k + r]).inserted {
                experts.append(guess[row * k + r])
                ranks.append(r)
            }
        }
        return (experts, ranks)
    }

    /// Routed experts' weighted sum; `indices` address `gateProj`/`upProj`/`downProj` (experts, or slots when streaming).
    func experts(_ x: MLXArray, _ indices: MLXArray, _ weights: MLXArray, fast: Bool, keepFP32: Bool = false) -> MLXArray {
        let (b, length, d) = (x.dim(0), x.dim(1), x.dim(2))
        if fast, b * length == 1, !keepFP32 {
            return fused(x: x, expertIDs: indices, weights: weights).reshaped([b, length, d])
        }
        let routes = indices.dim(-1)
        let out: MLXArray
        if indices.size >= 64 {
            // Sort routes by expert so each expert's rows are contiguous (matches `_gather_sort`).
            let flat = indices.flattened()
            let order = argSort(flat, axis: -1)
            let inverse = argSort(order, axis: -1)
            let rows = take(x.reshaped([b * length, 1, d]), floorDivide(order, Int32(routes)), axis: 0)  // integer division, like `//`
            let idx = take(flat, order, axis: 0)
            let up = upProj(rows, idx, sorted: true)
            let gate = gateProj(rows, idx, sorted: true)
            let down = downProj(activation(up: up, gate: gate), idx, sorted: true)
            out = take(down, inverse, axis: 0).reshaped([b, length, routes, d])
        } else {
            let rows = x.expandedDimensions(axes: [-2, -3])
            let up = upProj(rows, indices, sorted: false)
            let gate = gateProj(rows, indices, sorted: false)
            out = downProj(activation(up: up, gate: gate), indices, sorted: false).squeezed(axis: -2)
        }
        let sum = (out.asType(.float32) * weights.expandedDimensions(axis: -1)).sum(axis: -2)
        return keepFP32 ? sum : sum.asType(out.dtype)
    }
}
