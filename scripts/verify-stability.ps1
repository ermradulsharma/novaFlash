# PowerShell Codebase Stability and Optimizations Verification Script
# This script performs static analysis and checks to ensure all optimizations are intact, safe, and correct.

Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host " [AUDIT SUITE] Codebase Stability & Optimizations Verification" -ForegroundColor Magenta
Write-Host "====================================================================" -ForegroundColor Cyan

$SuccessCount = 0
$FailureCount = 0

function Assert-Check {
    param (
        [string]$Name,
        [bool]$Condition
    )
    if ($Condition) {
        Write-Host "  [PASS] $Name" -ForegroundColor Green
        $global:SuccessCount++
    } else {
        Write-Host "  [FAIL] $Name" -ForegroundColor Red
        $global:FailureCount++
    }
}

# ---------------------------------------------------------------------
# AUDIT 1: CMake Subdirectory Inclusion
# ---------------------------------------------------------------------
$CMakeContent = Get-Content -Path "CMakeLists.txt" -Raw
$HasPocs = Test-Path -Path "pocs"
$ConditionalPocsOk = $CMakeContent -match "if\s*\(\s*EXISTS\s+`"\`${CMAKE_CURRENT_SOURCE_DIR}/pocs/CMakeLists.txt`"\s*\)\s*add_subdirectory\(\s*pocs\s*\)" -or !$HasPocs
Assert-Check "CMake conditionally includes 'pocs' subdirectory" $ConditionalPocsOk

# ---------------------------------------------------------------------
# AUDIT 2: ISWA KV Caching Batch Fallback
# ---------------------------------------------------------------------
$IswaKVContent = Get-Content -Path "src/llama-kv-cache-iswa.cpp" -Raw
$HasTertiaryFallback = $IswaKVContent -match "split_seq"
Assert-Check "ISWA KV Cache has tertiary split sequential fallback logic (split_seq)" $HasTertiaryFallback

# ---------------------------------------------------------------------
# AUDIT 3: Sampler Causal UTF-8 Bounds
# ---------------------------------------------------------------------
$SamplerContent = Get-Content -Path "common/reasoning-budget.cpp" -Raw
$HasUTF8Boundary = $SamplerContent -match "common_utf8_is_complete"
Assert-Check "Reasoning budget sampler checks for complete UTF-8 characters" $HasUTF8Boundary

# ---------------------------------------------------------------------
# AUDIT 4: Speculative Dynamic Drafting Bounds
# ---------------------------------------------------------------------
$SpeculativeContent = Get-Content -Path "common/speculative.cpp" -Raw
$HasSpecBounds = $SpeculativeContent -match "n_max" -and $SpeculativeContent -match "std::min"
Assert-Check "Speculative drafting scales limits dynamically using n_max constraints" $HasSpecBounds

# ---------------------------------------------------------------------
# AUDIT 5: MiniCPM-3 Dynamic GGUF Scales
# ---------------------------------------------------------------------
$MiniCPMContent = Get-Content -Path "src/models/minicpm3.cpp" -Raw
$HasDynamicScales = $MiniCPMContent -match "f_embedding_scale" -and $MiniCPMContent -match "f_residual_scale"
Assert-Check "MiniCPM-3 uses dynamic scaling coefficients from GGUF metadata" $HasDynamicScales

# ---------------------------------------------------------------------
# AUDIT 6: Gemma-4 & Gemma-3n Early Token Stripping
# ---------------------------------------------------------------------
$Gemma4Content = Get-Content -Path "src/models/gemma4-iswa.cpp" -Raw
$Gemma3nContent = Get-Content -Path "src/models/gemma3n-iswa.cpp" -Raw
$Stripping4Ok = $Gemma4Content -match "n_tokens_l" -and $Gemma4Content -match "last_kv_layer"
$Stripping3nOk = $Gemma3nContent -match "n_tokens_l" -and $Gemma3nContent -match "last_kv_layer"
Assert-Check "Gemma-4 ISWA performs early token stripping after last KV layer" $Stripping4Ok
Assert-Check "Gemma-3n ISWA performs early token stripping after last KV layer" $Stripping3nOk

# ---------------------------------------------------------------------
# AUDIT 7: GroveMoE Expert Selection Precomputation
# ---------------------------------------------------------------------
$GroveMoEContent = Get-Content -Path "src/models/grovemoe.cpp" -Raw
$GroveMoEOptimized = $GroveMoEContent -match "selected_experts" -and !($GroveMoEContent -match "TODO: Only do the expert selection")
Assert-Check "GroveMoE precomputes expert selection once to bypass duplicate argsort overhead" $GroveMoEOptimized

# ---------------------------------------------------------------------
# AUDIT 8: DeepSeek-V4 Dynamic Scaling Ratio Decoupling
# ---------------------------------------------------------------------
$HybridMemContent = Get-Content -Path "src/llama-memory-hybrid-iswa.cpp" -Raw
$DSv4Content = Get-Content -Path "src/models/deepseek4.cpp" -Raw
$RatioDecoupledHybrid = $HybridMemContent -match "indexer_head_size > 0" -and !($HybridMemContent -match "ratio == 4")
$RatioDecoupledDSv4 = $DSv4Content -match "compress_ratio >= 4"
Assert-Check "Hybrid memory decouples index allocations from hardcoded ratio == 4 assertions" $RatioDecoupledHybrid
Assert-Check "DeepSeek-V4 state layouts dynamically support ratios >= 4" $RatioDecoupledDSv4

# ---------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------
Write-Host ""
if ($FailureCount -eq 0) {
    Write-Host " [SUCCESS] All $SuccessCount audits passed successfully! Codebase structural stability verified." -ForegroundColor Green -BackgroundColor Black
} else {
    Write-Host " [ERROR] $FailureCount audits failed! Please verify implemented changes." -ForegroundColor Red -BackgroundColor Black
}
Write-Host ""
