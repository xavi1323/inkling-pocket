// Metal sources for the single-token MoE (fast path). Affine-quantized rows are a little-endian bit stream:
// value j sits at bits [j*BITS, j*BITS+BITS), for every width 2–8, so 8 values always fill exactly BITS bytes.

enum FusedMoESource {
    /// One simdgroup per (expert slot, intermediate row): gate and up dot products, then the softcapped SwiGLU.
    /// Grid (32*I, K, 1). Template: T, D, I, GB, GG, UB, UG.
    static let gateUp = """
        uint gid = thread_position_in_grid.x;
        uint r = gid / 32, lane = gid % 32;
        uint k = thread_position_in_grid.y;
        if (r >= (uint)I) return;
        size_t row = (size_t)ids[k] * I + r;
        const device uint8_t* gp = (const device uint8_t*)gw + row * (D * GB / 8);
        const device uint8_t* up = (const device uint8_t*)uw + row * (D * UB / 8);
        size_t gsb = row * (D / GG), usb = row * (D / UG);
        float ga = 0.0f, ua = 0.0f;
        for (uint c = lane; c < (uint)(D / 8); c += 32) {
            float xv[8]; float xs = 0.0f;
            for (int j = 0; j < 8; ++j) { xv[j] = float(x[c * 8 + j]); xs += xv[j]; }
            ulong gv = 0, uv = 0;
            for (int b = 0; b < GB; ++b) gv |= ulong(gp[c * GB + b]) << (8 * b);
            for (int b = 0; b < UB; ++b) uv |= ulong(up[c * UB + b]) << (8 * b);
            float gq = 0.0f, uq = 0.0f;
            for (int j = 0; j < 8; ++j) {
                gq += float((gv >> (j * GB)) & ((1ul << GB) - 1)) * xv[j];
                uq += float((uv >> (j * UB)) & ((1ul << UB) - 1)) * xv[j];
            }
            uint g = (c * 8) / GG, u = (c * 8) / UG;
            ga += float(gs[gsb + g]) * gq + float(gbias[gsb + g]) * xs;
            ua += float(us[usb + u]) * uq + float(ubias[usb + u]) * xs;
        }
        ga = simd_sum(ga);
        ua = simd_sum(ua);
        if (lane == 0) {
            float g = float(T(ga)), u = float(T(ua));
            float gc = caps[0], uc = caps[1];
            float a = precise::tanh(g / gc) * gc * (1.0f / (1.0f + precise::exp(-g))) * precise::tanh(u / uc) * uc;
            h[k * I + r] = T(a);
        }
        """

    /// One simdgroup per output row: down dot product for each routed expert, weighted sum in FP32.
    /// Grid (32*D, 1, 1). Template: T, D, I, K, DB, DG.
    static let down = """
        uint gid = thread_position_in_grid.x;
        uint r = gid / 32, lane = gid % 32;
        if (r >= (uint)D) return;
        float total = 0.0f;
        for (int k = 0; k < K; ++k) {
            size_t row = (size_t)ids[k] * D + r;
            const device uint8_t* p = (const device uint8_t*)dw + row * (I * DB / 8);
            size_t sb = row * (I / DG);
            float acc = 0.0f;
            for (uint c = lane; c < (uint)(I / 8); c += 32) {
                float hv[8]; float hs = 0.0f;
                for (int j = 0; j < 8; ++j) { hv[j] = float(h[k * I + c * 8 + j]); hs += hv[j]; }
                ulong v = 0;
                for (int b = 0; b < DB; ++b) v |= ulong(p[c * DB + b]) << (8 * b);
                float q = 0.0f;
                for (int j = 0; j < 8; ++j) q += float((v >> (j * DB)) & ((1ul << DB) - 1)) * hv[j];
                uint g = (c * 8) / DG;
                acc += float(ds[sb + g]) * q + float(dbias[sb + g]) * hs;
            }
            acc = simd_sum(acc);
            total += float(T(acc)) * rw[k];
        }
        if (lane == 0) out[r] = T(total);
        """
}
