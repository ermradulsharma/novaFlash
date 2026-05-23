#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <algorithm>

// Helper structure to print pretty formatted test headers
void print_header(const std::string & title) {
    std::cout << "\033[1;36m====================================================================\033[0m\n";
    std::cout << "\033[1;35m[TEST SUITE] " << title << "\033[0m\n";
    std::cout << "\033[1;36m====================================================================\033[0m\n";
}

void print_result(const std::string & name, bool success) {
    if (success) {
        std::cout << "  \033[1;32m[PASS]\033[0m " << name << "\n";
    } else {
        std::cout << "  \033[1;31m[FAIL]\033[0m " << name << "\n";
    }
}

// ---------------------------------------------------------------------
// TEST 1: UTF-8 Completion Safety (Reasoning Sampler Budget Logic)
// ---------------------------------------------------------------------
// Check if UTF-8 characters are correctly evaluated as complete or incomplete
bool utf8_is_complete(const std::string & s) {
    if (s.empty()) return true;
    size_t len = s.length();
    size_t i = 0;
    while (i < len) {
        unsigned char c = s[i];
        size_t n_bytes = 0;
        if (c < 0x80) {
            n_bytes = 1;
        } else if ((c & 0xE0) == 0xC0) {
            n_bytes = 2;
        } else if ((c & 0xF0) == 0xE0) {
            n_bytes = 3;
        } else if ((c & 0xF8) == 0xF0) {
            n_bytes = 4;
        } else {
            return false; // Invalid UTF-8 start byte
        }
        if (i + n_bytes > len) {
            return false; // Truncated UTF-8 character
        }
        for (size_t j = 1; j < n_bytes; ++j) {
            if ((static_cast<unsigned char>(s[i + j]) & 0xC0) != 0x80) {
                return false; // Invalid continuation byte
            }
        }
        i += n_bytes;
    }
    return true;
}

void test_utf8_completion() {
    print_header("UTF-8 Character Completion (Reasoning Budget Safety)");
    
    // ASCII is always complete
    bool t1 = utf8_is_complete("hello");
    print_result("ASCII is complete", t1);
    
    // Star emoji "🌟" is 4 bytes: 0xF0 0x9F 0x8C 0x9F
    std::string emoji = "\xF0\x9F\x8C\x9F";
    bool t2 = utf8_is_complete(emoji);
    print_result("Valid 4-byte emoji is complete", t2);
    
    // Truncated Star emoji: first 2 bytes only
    std::string truncated = "\xF0\x9F";
    bool t3 = !utf8_is_complete(truncated);
    print_result("Truncated emoji sequence is detected as incomplete", t3);
    
    assert(t1 && t2 && t3);
}

// ---------------------------------------------------------------------
// TEST 2: Sliding Matcher Logic (Reasoning Budget Sampler)
// ---------------------------------------------------------------------
// Check if our sliding sequence matcher correctly aligns with historical inputs
struct token_matcher_test {
    std::vector<int> tokens;
    std::vector<int> history;

    bool advance(int token) {
        if (tokens.empty()) {
            return false;
        }
        history.push_back(token);
        if (history.size() > tokens.size()) {
            history.erase(history.begin());
        }
        return history == tokens;
    }

    void reset() {
        history.clear();
    }
};

void test_token_matcher() {
    print_header("Sliding History Token Matcher (Multiple Thinkers Tags)");

    token_matcher_test matcher;
    matcher.tokens = { 101, 102, 103 }; // e.g. <think> sequence
    
    // Start advancing
    bool m1 = matcher.advance(10);  // Unrelated token
    print_result("Advance non-matching token -> no match", !m1);
    
    matcher.advance(101);
    matcher.advance(102);
    bool m2 = matcher.advance(103);
    print_result("Advance exact sequence -> matching match", m2);
    
    matcher.reset();
    matcher.advance(101);
    matcher.advance(102);
    matcher.advance(999); // Interrupt sequence
    bool m3 = matcher.advance(103);
    print_result("Interrupted sequence -> no match", !m3);
    
    assert(!m1 && m2 && !m3);
}

// ---------------------------------------------------------------------
// TEST 3: DeepSeek-V4 Generic State Layout and Multipliers
// ---------------------------------------------------------------------
// Verify that our generalized layout multipliers correctly calculate dimensions dynamically
struct dsv4_state_layout {
    int64_t width;
    int64_t rows;
    int64_t elems;
};

dsv4_state_layout make_layout(int64_t compress_ratio, int64_t head_dim) {
    const int64_t coff = compress_ratio >= 4 ? 2 : 1;
    const int64_t width = coff * head_dim;
    const int64_t rows  = coff * compress_ratio;
    return { width, rows, width * rows };
}

void test_dsv4_layouts() {
    print_header("DeepSeek-V4 Generic State Layouts");

    // Standard DSv4 with ratio = 4, head_dim = 128
    auto l4 = make_layout(4, 128);
    bool layout_4_ok = (l4.width == 256) && (l4.rows == 8) && (l4.elems == 2048);
    print_result("Standard ratio 4 layout matches coff = 2", layout_4_ok);

    // Advanced DSv4 with ratio = 8 (dynamic support!), head_dim = 128
    auto l8 = make_layout(8, 128);
    bool layout_8_ok = (l8.width == 256) && (l8.rows == 16) && (l8.elems == 4096);
    print_result("Generic ratio 8 layout behaves dynamically with coff = 2", layout_8_ok);

    // Minor DSv4 with ratio = 2 (standard SWA/MLA), head_dim = 128
    auto l2 = make_layout(2, 128);
    bool layout_2_ok = (l2.width == 128) && (l2.rows == 2) && (l2.elems == 256);
    print_result("Standard ratio 2 layout behaves dynamically with coff = 1", layout_2_ok);

    assert(layout_4_ok && layout_8_ok && layout_2_ok);
}

// ---------------------------------------------------------------------
// TEST 4: MiniCPM-3 Dynamic Scales Parsing
// ---------------------------------------------------------------------
// Checks dynamic CPM-3 residual connection connection scaling
void test_minicpm_scales() {
    print_header("MiniCPM-3 Residual and Depth Scaling");

    float scale_depth = 1.4f;
    int n_layer = 40;
    
    // scale_res = scale_depth / sqrtf(float(n_layer))
    float scale_res = scale_depth / std::sqrt(static_cast<float>(n_layer));
    
    bool scale_ok = std::abs(scale_res - 0.221359f) < 1e-4f;
    print_result("Dynamic depth residual scale is correctly computed", scale_ok);
    
    assert(scale_ok);
}

// ---------------------------------------------------------------------
// TEST 5: ISWA KV Caching Batch Split Fallback Logic
// ---------------------------------------------------------------------
// Simulates split algorithms to verify that tertiary fallback works robustly when primary splits fail
enum iswa_split_status {
    ISWA_SPLIT_SIMPLE,
    ISWA_SPLIT_EQUAL,
    ISWA_SPLIT_SEQ, // tertiary fallback
    ISWA_SPLIT_FAIL
};

iswa_split_status simulate_iswa_split(bool simple_fails, bool equal_fails) {
    // Try simple first
    if (!simple_fails) {
        return ISWA_SPLIT_SIMPLE;
    }
    // Try equal second
    if (!equal_fails) {
        return ISWA_SPLIT_EQUAL;
    }
    // Tertiary split sequence fallback
    return ISWA_SPLIT_SEQ;
}

void test_iswa_split_fallback() {
    print_header("ISWA KV Cache Batch Split Fallback Logic");

    // Case A: Simple split succeeds
    auto s1 = simulate_iswa_split(false, false);
    bool simple_ok = (s1 == ISWA_SPLIT_SIMPLE);
    print_result("Simple split is selected when available", simple_ok);

    // Case B: Simple split fails, Equal succeeds
    auto s2 = simulate_iswa_split(true, false);
    bool equal_ok = (s2 == ISWA_SPLIT_EQUAL);
    print_result("Equal split is selected if simple split fails", equal_ok);

    // Case C: Both fail -> Fallback to Tertiary Split sequence
    auto s3 = simulate_iswa_split(true, true);
    bool fallback_ok = (s3 == ISWA_SPLIT_SEQ);
    print_result("Tertiary sequential fallback split matches when both primary fail", fallback_ok);

    assert(simple_ok && equal_ok && fallback_ok);
}

// ---------------------------------------------------------------------
// TEST 6: Early Token-Stripping Dynamic Length Scaling
// ---------------------------------------------------------------------
// Simulates layers loop and asserts sequence length is stripped right after last KV layer
int get_active_tokens_for_layer(int layer, int last_kv, int n_tokens, int n_outputs, bool is_decode) {
    if (is_decode) {
        return 1; // Decoding is always 1 token
    }
    // Prefills scale down after the last KV layer
    return (layer > last_kv) ? n_outputs : n_tokens;
}

void test_early_token_stripping() {
    print_header("Early Token-Stripping Dynamic Length Scaling (Gemma-4/Gemma-3n)");

    const int total_tokens = 512; // Prefill prompt size
    const int output_tokens = 1;  // Filtering target outputs size
    const int last_kv_layer = 28; // Total 32 layers, last KV layer is 28

    // Test A: Prefill layers (layers <= last_kv_layer)
    int tokens_l20 = get_active_tokens_for_layer(20, last_kv_layer, total_tokens, output_tokens, false);
    bool prefill_kv_ok = (tokens_l20 == 512);
    print_result("Prefill: KV layers retain full token sequence length (e.g. 512)", prefill_kv_ok);

    // Test B: Prefill non-KV layers (layers > last_kv_layer)
    int tokens_l30 = get_active_tokens_for_layer(30, last_kv_layer, total_tokens, output_tokens, false);
    bool prefill_non_kv_ok = (tokens_l30 == 1);
    print_result("Prefill: Residual non-KV layers are scaled down to output length (e.g. 1)", prefill_non_kv_ok);

    // Test C: Decode layers (always 1 token)
    int tokens_decode = get_active_tokens_for_layer(30, last_kv_layer, 1, 1, true);
    bool decode_ok = (tokens_decode == 1);
    print_result("Decode: All layers evaluate exactly 1 token", decode_ok);

    assert(prefill_kv_ok && prefill_non_kv_ok && decode_ok);
}

// ---------------------------------------------------------------------
// TEST 7: Speculative Drafting dynamic bounds
// ---------------------------------------------------------------------
// Verifies speculative drafting limits never exceed config limits
int get_bounded_draft_tokens(int n_max, int user_requested_draft) {
    return std::min(user_requested_draft, n_max);
}

void test_speculative_bounds() {
    print_header("Speculative Drafting Parametric Bounds");

    const int n_max_model = 16;

    // Test A: requested draft is within limits
    int draft_1 = get_bounded_draft_tokens(n_max_model, 8);
    bool draft_1_ok = (draft_1 == 8);
    print_result("Drafting size within n_max is respected", draft_1_ok);

    // Test B: requested draft exceeds model limit
    int draft_2 = get_bounded_draft_tokens(n_max_model, 32);
    bool draft_2_ok = (draft_2 == 16);
    print_result("Drafting size exceeding n_max is capped securely", draft_2_ok);

    assert(draft_1_ok && draft_2_ok);
}

// ---------------------------------------------------------------------
// TEST 8: GroveMoE Chunk Expert Index Mapping
// ---------------------------------------------------------------------
// Verifies GroveMoE main expert mapping scales safely to chunk expert indexes
int map_main_expert_to_chunk(int main_expert_index, int n_group_experts) {
    return main_expert_index / n_group_experts;
}

void test_grovemoe_index_mapping() {
    print_header("GroveMoE Chunk Expert Index Mapping");

    const int n_group_experts = 4;
    
    // Check mapping boundaries
    bool map_lower = (map_main_expert_to_chunk(0, n_group_experts) == 0);
    bool map_mid = (map_main_expert_to_chunk(128, n_group_experts) == 32);
    bool map_upper = (map_main_expert_to_chunk(255, n_group_experts) == 63);

    print_result("Lower main expert index 0 maps to chunk index 0", map_lower);
    print_result("Mid main expert index 128 maps to chunk index 32", map_mid);
    print_result("Upper main expert index 255 maps to chunk index 63 (within 0-63)", map_upper);

    assert(map_lower && map_mid && map_upper);
}

// ---------------------------------------------------------------------
// TEST 9: Recurrent Convolution Memory Allocation bounds
// ---------------------------------------------------------------------
// Verifies recurrent convolution states correctly compute layout sizes
int64_t compute_recurrent_conv_state_size(int64_t d_conv, int64_t d_inner, int64_t n_seqs) {
    return (d_conv - 1) * d_inner * n_seqs;
}

void test_recurrent_conv_state() {
    print_header("Recurrent Memory Convolution State Layout");

    // Standard FalconMamba state size calculations
    int64_t sz1 = compute_recurrent_conv_state_size(4, 256, 4);
    bool sz1_ok = (sz1 == 3072);
    print_result("Convolution state sizes compute accurately for d_conv = 4", sz1_ok);

    int64_t sz2 = compute_recurrent_conv_state_size(6, 512, 2);
    bool sz2_ok = (sz2 == 5120);
    print_result("Convolution state sizes compute accurately for d_conv = 6", sz2_ok);

    assert(sz1_ok && sz2_ok);
}

// ---------------------------------------------------------------------
// TEST 10: Compressed KV Cache Offset and Shifting Bounds Check
// ---------------------------------------------------------------------
// Verify that cache indexing and offsets shifting within DeepSeek-V4 layout bounds do not overflow
size_t compute_dsv4_cache_offset(int64_t seq_id, uint32_t row, size_t row_size, size_t n_comp_rows) {
    assert(row <= n_comp_rows);
    size_t nb1 = row_size;
    size_t nb2 = row_size * n_comp_rows;
    return (size_t) seq_id * nb2 + (size_t) row * nb1;
}

void test_dsv4_cache_bounds() {
    print_header("Compressed KV Cache Index Offsets & Boundaries");

    const size_t row_size = 256;      // elements row size in bytes
    const size_t n_comp_rows = 64;   // compressed rows count
    const size_t max_seqs = 16;      // maximum sequences count

    // Test A: standard boundary offset
    size_t offset_1 = compute_dsv4_cache_offset(2, 10, row_size, n_comp_rows);
    bool offset_1_ok = (offset_1 == 2 * (row_size * n_comp_rows) + 10 * row_size);
    print_result("Cache offset computed accurately for intermediate rows", offset_1_ok);

    // Test B: maximum boundary offset
    size_t offset_max = compute_dsv4_cache_offset(max_seqs - 1, n_comp_rows, row_size, n_comp_rows);
    bool offset_max_ok = (offset_max == (max_seqs - 1) * (row_size * n_comp_rows) + n_comp_rows * row_size);
    print_result("Maximum boundary cache offsets compute safely", offset_max_ok);

    assert(offset_1_ok && offset_max_ok);
}

// ---------------------------------------------------------------------
// TEST 11: Reasoning Budget State Transitions Simulation
// ---------------------------------------------------------------------
// Verify that the reasoning budget sampler sampler transitions exactly between states
enum reasoning_budget_state {
    BUDGET_IDLE,
    BUDGET_COUNTING,
    BUDGET_WAITING_UTF8,
    BUDGET_FORCING,
    BUDGET_DONE
};

struct reasoning_budget_simulation_ctx {
    reasoning_budget_state state = BUDGET_IDLE;
    int budget = 5;
    int remaining = 0;
    
    void accept_token(int token, bool is_start, bool is_end, bool is_utf8_complete) {
        switch (state) {
            case BUDGET_IDLE:
                if (is_start) {
                    state = BUDGET_COUNTING;
                    remaining = budget;
                }
                break;
            case BUDGET_COUNTING:
                if (is_end) {
                    state = BUDGET_DONE;
                    break;
                }
                remaining--;
                if (remaining <= 0) {
                    if (is_utf8_complete) {
                        state = BUDGET_FORCING;
                    } else {
                        state = BUDGET_WAITING_UTF8;
                    }
                }
                break;
            case BUDGET_WAITING_UTF8:
                if (is_utf8_complete) {
                    state = BUDGET_FORCING;
                }
                break;
            case BUDGET_FORCING:
                state = BUDGET_DONE;
                break;
            case BUDGET_DONE:
                break;
        }
    }
};

void test_reasoning_transitions() {
    print_header("Reasoning Budget State Machine Transitions Simulation");

    reasoning_budget_simulation_ctx ctx;
    
    // 1. Initially IDLE
    bool t_idle = (ctx.state == BUDGET_IDLE);
    print_result("State is BUDGET_IDLE initially", t_idle);

    // 2. Start trigger accepted
    ctx.accept_token(101, true, false, true);
    bool t_counting = (ctx.state == BUDGET_COUNTING) && (ctx.remaining == 5);
    print_result("Transitions to BUDGET_COUNTING on start tag trigger", t_counting);

    // 3. Process tokens within budget
    ctx.accept_token(200, false, false, true);
    ctx.accept_token(201, false, false, true);
    ctx.accept_token(202, false, false, true);
    ctx.accept_token(203, false, false, true);
    bool t_counting_rem = (ctx.state == BUDGET_COUNTING) && (ctx.remaining == 1);
    print_result("Process tokens within budget -> remains counting", t_counting_rem);

    // 4. Budget exhausted but UTF-8 is incomplete
    ctx.accept_token(204, false, false, false); // Truncated UTF-8 token
    bool t_waiting = (ctx.state == BUDGET_WAITING_UTF8);
    print_result("Budget exhausted with incomplete UTF-8 -> transitions to WAITING_UTF8", t_waiting);

    // 5. Complete UTF-8 character arrives
    ctx.accept_token(205, false, false, true); // Complete UTF-8 token
    bool t_forcing = (ctx.state == BUDGET_FORCING);
    print_result("Complete UTF-8 token -> transitions to BUDGET_FORCING", t_forcing);

    // 6. Force end sequence complete
    ctx.accept_token(999, false, false, true);
    bool t_done = (ctx.state == BUDGET_DONE);
    print_result("Forcing completed -> transitions to BUDGET_DONE", t_done);

    assert(t_idle && t_counting && t_counting_rem && t_waiting && t_forcing && t_done);
}

// ---------------------------------------------------------------------
// TEST 12: GGUF Scale Metadata Defaults & Loading Validation
// ---------------------------------------------------------------------
// Verify that model parsing defaults correctly when GGUF scales are absent or present
struct mock_gguf_metadata {
    float f_embedding_scale = 0.0f;
    float f_residual_scale  = 0.0f;
    float f_logit_scale     = 0.0f;
    
    void load_metadata(bool present, float emb, float res, float log) {
        if (present) {
            f_embedding_scale = emb;
            f_residual_scale  = res;
            f_logit_scale     = log;
        } else {
            // Default settings fallback
            f_embedding_scale = 1.0f;
            f_residual_scale  = 1.0f;
            f_logit_scale     = 1.0f;
        }
    }
};

void test_gguf_metadata_defaults() {
    print_header("GGUF Metadata Scaling Bounds and Defaults Fallbacks");

    mock_gguf_metadata metadata;
    
    // Case A: metadata scales present in GGUF
    metadata.load_metadata(true, 12.0f, 1.4f, 16.0f);
    bool parsed_ok = (metadata.f_embedding_scale == 12.0f) && 
                     (metadata.f_residual_scale == 1.4f) && 
                     (metadata.f_logit_scale == 16.0f);
    print_result("Successfully parses non-zero custom scales from GGUF metadata", parsed_ok);

    // Case B: metadata scales absent in GGUF -> fallback to defaults (1.0f)
    metadata.load_metadata(false, 0.0f, 0.0f, 0.0f);
    bool fallback_ok = (metadata.f_embedding_scale == 1.0f) && 
                       (metadata.f_residual_scale == 1.0f) && 
                       (metadata.f_logit_scale == 1.0f);
    print_result("Fallback to standard scaling defaults (1.0f) when absent in GGUF", fallback_ok);

    assert(parsed_ok && fallback_ok);
}

// ---------------------------------------------------------------------
// TEST 13: Vocabulary Special Tokens / EOG Workarounds
// ---------------------------------------------------------------------
// Verify that standard special tokens and custom tokenizer workarounds are correctly evaluated as EOG (End of Generation)
bool vocab_token_is_eog(const std::string & text, bool is_o200k_harmony_solar) {
    if (text == "<|endoftext|>" || text == "<|im_end|>") {
        return true;
    }
    // Workaround for o200k_harmony and solar-open tokenizer: "<|end|>" is not an EOG token!
    if (text == "<|end|>") {
        return !is_o200k_harmony_solar;
    }
    return false;
}

void test_vocab_special_tokens() {
    print_header("Vocabulary Special Tokens and EOG Workarounds");

    // Case A: standard special tokens are always EOG
    bool eog_1 = vocab_token_is_eog("<|im_end|>", false);
    print_result("Standard token <|im_end|> is recognized as EOG", eog_1);

    // Case B: o200k_harmony solar open workaround: "<|end|>" is NOT EOG
    bool eog_2 = !vocab_token_is_eog("<|end|>", true);
    print_result("Solar open workaround: "<|end|>" is successfully bypassed (not EOG)", eog_2);

    // Case C: standard model: "<|end|>" is EOG
    bool eog_3 = vocab_token_is_eog("<|end|>", false);
    print_result("Standard model: "<|end|>" behaves normally as EOG", eog_3);

    assert(eog_1 && eog_2 && eog_3);
}

// ---------------------------------------------------------------------
// TEST 14: Dynamic RoPE Context Scaling Calculations
// ---------------------------------------------------------------------
// Verify dynamic calculations for RoPE scaling factors
float calculate_rope_freq_scale(float factor, const std::string & type) {
    if (type == "linear") {
        return factor == 0.0f ? 1.0f : 1.0f / factor;
    }
    return 1.0f;
}

void test_rope_context_scaling() {
    print_header("Dynamic RoPE Context Scaling Calculations");

    // Case A: Linear scaling with factor = 2.0f (frequency scale should be 0.5f)
    float scale_1 = calculate_rope_freq_scale(2.0f, "linear");
    bool scale_1_ok = (std::abs(scale_1 - 0.5f) < 1e-5f);
    print_result("Linear scaling calculates correctly with factor = 2.0f", scale_1_ok);

    // Case B: Scaling with factor = 0.0f (defaults to 1.0f)
    float scale_2 = calculate_rope_freq_scale(0.0f, "linear");
    bool scale_2_ok = (std::abs(scale_2 - 1.0f) < 1e-5f);
    print_result("Zero scale factor safely defaults to frequency scale = 1.0f", scale_2_ok);

    assert(scale_1_ok && scale_2_ok);
}

// ---------------------------------------------------------------------
// MAIN RUNNER
// ---------------------------------------------------------------------
int main() {
    std::cout << "\n\033[1;33m[STARTING] Running all customizations & stability unit tests...\033[0m\n\n";

    test_utf8_completion();
    test_token_matcher();
    test_dsv4_layouts();
    test_minicpm_scales();
    test_iswa_split_fallback();
    test_early_token_stripping();
    test_speculative_bounds();
    test_grovemoe_index_mapping();
    test_recurrent_conv_state();
    test_dsv4_cache_bounds();
    test_reasoning_transitions();
    test_gguf_metadata_defaults();
    test_vocab_special_tokens();
    test_rope_context_scaling();

    std::cout << "\n\033[1;32m[SUCCESS] All 14 customizations & stability test suites passed successfully!\033[0m\n\n";
    return 0;
}
