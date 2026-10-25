"""BPE on-bolme icin Unicode sinif tablolari (\\p{L}, \\p{N}) uretir.

Kullanim: python gen_unicode_tables.py > include/nexusdata/text/unicode_tables.inc
Kaynak: Python unicodedata (Unicode surumu dosyaya yazilir). tiktoken'in Rust regex'i de
Unicode 16 kullanir; surum farki olursa nadir karakterlerde ayrisma beklenir (raporlanir).
"""
import sys
import unicodedata


def ranges(pred):
    out = []
    start = None
    for cp in range(0x110000):
        ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def emit(name, rs):
    print(f"inline constexpr std::uint32_t {name}[][2] = {{")
    for i in range(0, len(rs), 4):
        print("    " + " ".join(f"{{0x{a:X}, 0x{b:X}}}," for a, b in rs[i:i + 4]))
    print("};")


letters = ranges(lambda c: unicodedata.category(c).startswith("L"))
numbers = ranges(lambda c: unicodedata.category(c).startswith("N"))
print(f"// Uretildi: tools/gen_unicode_tables.py, Unicode {unicodedata.unidata_version}. Elle duzenlemeyin.")
print(f"inline constexpr const char* kUnicodeVersion = \"{unicodedata.unidata_version}\";")
emit("kLetterRanges", letters)
emit("kNumberRanges", numbers)
sys.stderr.write(f"L: {len(letters)} aralik, N: {len(numbers)} aralik\n")
