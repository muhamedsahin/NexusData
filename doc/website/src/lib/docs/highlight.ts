import { codeToHtml } from "shiki";

const LANG_ALIASES: Record<string, string> = {
  cpp: "cpp",
  "c++": "cpp",
  cmake: "cmake",
  bash: "bash",
  sh: "bash",
  shell: "bash",
  json: "json",
  python: "python",
  py: "python",
  powershell: "powershell",
  ps1: "powershell",
  ts: "typescript",
  tsx: "tsx",
  js: "javascript",
  text: "text",
};

export async function highlightCode(
  code: string,
  lang = "text",
  options?: { highlightLines?: number[] },
): Promise<string> {
  const language = LANG_ALIASES[lang.toLowerCase()] ?? lang;
  try {
    return await codeToHtml(code, {
      lang: language,
      theme: "github-dark-default",
      transformers: options?.highlightLines?.length
        ? [
            {
              line(node, line) {
                if (options.highlightLines?.includes(line)) {
                  this.addClassToHast(node, "line-highlight");
                }
              },
            },
          ]
        : undefined,
    });
  } catch {
    return await codeToHtml(code, {
      lang: "text",
      theme: "github-dark-default",
    });
  }
}
