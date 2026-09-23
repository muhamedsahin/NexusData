export function detectWebGL(): boolean {
  if (typeof document === "undefined") return false;
  try {
    const canvas = document.createElement("canvas");
    const gl =
      canvas.getContext("webgl2", { failIfMajorPerformanceCaveat: true }) ??
      canvas.getContext("webgl", { failIfMajorPerformanceCaveat: true }) ??
      canvas.getContext("experimental-webgl");
    return Boolean(gl);
  } catch {
    return false;
  }
}

export function prefersReducedMotion(): boolean {
  if (typeof window === "undefined") return false;
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export const CMAKE_SNIPPET = `include(FetchContent)
FetchContent_Declare(
  matrixdata
  GIT_REPOSITORY https://github.com/PLACEHOLDER_ORG/matrixdata.git
  GIT_TAG        v1.0.0
)
FetchContent_MakeAvailable(matrixdata)
target_link_libraries(my_app PRIVATE matrixdata::matrixdata)`;
