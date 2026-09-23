"use client";

import { useMemo, useState } from "react";
import { useLocale } from "next-intl";
import { StatusBadge } from "@/components/ui/status-badge";

type Os = "linux" | "macos" | "windows";
type Pm = "cmake" | "vcpkg" | "conan";

export function InstallCommandGenerator() {
  const locale = useLocale() === "tr" ? "tr" : "en";
  const [os, setOs] = useState<Os>("linux");
  const [pm, setPm] = useState<Pm>("cmake");
  const [cuda, setCuda] = useState(false);

  const snippet = useMemo(() => {
    const cudaFlag = cuda ? " -DNEXUSDATA_WITH_CUDA=ON" : "";
    if (pm === "vcpkg") {
      return `# vcpkg (manifest mode)\nvcpkg install\ncmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake${cudaFlag}\ncmake --build build`;
    }
    if (pm === "conan") {
      return `# Conan\nconan install . -of build\ncmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=build/conan_toolchain.cmake${cudaFlag}\ncmake --build build`;
    }
    const prefix =
      os === "windows"
        ? "cmake -S . -B build -G \"Ninja\" -DCMAKE_BUILD_TYPE=Release"
        : "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release";
    return `${prefix}${cudaFlag}\ncmake --build build -j\nctest --test-dir build --output-on-failure`;
  }, [os, pm, cuda]);

  return (
    <div className="my-6 rounded-2xl border border-[color:var(--border)] bg-[color:var(--surface)]/50 p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h3 className="font-semibold text-[color:var(--fg)]">
          {locale === "tr" ? "Kurulum komutu üretici" : "Install command generator"}
        </h3>
        <StatusBadge status="stable" label="v1.0" />
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">OS</span>
          <select
            value={os}
            onChange={(e) => setOs(e.target.value as Os)}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 text-sm"
          >
            <option value="linux">Linux</option>
            <option value="macos">macOS</option>
            <option value="windows">Windows</option>
          </select>
        </label>
        <label className="text-xs">
          <span className="mb-1 block text-[color:var(--fg-muted)]">
            {locale === "tr" ? "Paket" : "Package"}
          </span>
          <select
            value={pm}
            onChange={(e) => setPm(e.target.value as Pm)}
            className="w-full rounded-md border border-[color:var(--border)] bg-[color:var(--bg)] px-2 py-1.5 text-sm"
          >
            <option value="cmake">CMake</option>
            <option value="vcpkg">vcpkg</option>
            <option value="conan">Conan</option>
          </select>
        </label>
        <label className="flex items-end gap-2 pb-2 text-xs text-[color:var(--fg)]">
          <input type="checkbox" checked={cuda} onChange={(e) => setCuda(e.target.checked)} />
          CUDA (`NEXUSDATA_WITH_CUDA`)
        </label>
      </div>

      <pre className="mt-4 overflow-x-auto rounded-xl bg-[#0d1117] p-3 font-mono text-xs text-white/85">
        {snippet}
      </pre>
    </div>
  );
}
