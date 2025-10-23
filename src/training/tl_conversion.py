from pathlib import Path
import treelite
import tl2cgen

MODEL_JSON = Path("data/mod_latest.json")   # your existing XGBoost JSON
OUT_LIB    = Path("data/mod_latest.so")     # .dll on Windows, .dylib on macOS

# 1) Load XGBoost model into Treelite
#    (Treelite 4.x is the model exchange/serialization layer)
model = treelite.frontend.load_xgboost_model(str(MODEL_JSON))

# 2) Export a compiled shared library with TL2cgen
#    Use 'gcc' on Linux, 'clang' on macOS, 'msvc' on Windows
tl2cgen.export_lib(
    model,
    toolchain="gcc",
    libpath=str(OUT_LIB),
    params={"parallel_comp": 1},  # compile in parallel; tweak for your cores
    verbose=True,
)

print(f"✅ Exported: {OUT_LIB}")
