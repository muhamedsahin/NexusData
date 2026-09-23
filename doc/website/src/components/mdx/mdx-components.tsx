import type { ReactNode } from "react";
import { Callout } from "@/components/ui/callout";
import { StatusBadge } from "@/components/ui/status-badge";
import { highlightCode } from "@/lib/docs/highlight";
import { CodeBlock } from "@/components/mdx/code-block";
import { cn } from "@/lib/cn";
import { BenchmarkSuite } from "@/components/benchmarks/benchmark-suite";
import { DataLoaderSimulator } from "@/components/docs/interactive/dataloader-simulator";
import { SplitVisualizer } from "@/components/docs/interactive/split-visualizer";
import { PipelineBuilder } from "@/components/docs/interactive/pipeline-builder";
import { InstallCommandGenerator } from "@/components/docs/interactive/install-command-generator";
import { ShapeExplorer } from "@/components/docs/interactive/shape-explorer";

function Heading({
  as: Tag,
  id,
  children,
  className,
}: {
  as: "h2" | "h3" | "h4";
  id?: string;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <Tag
      id={id}
      className={cn(
        "scroll-mt-24 font-[family-name:var(--font-display)] font-semibold tracking-tight text-[color:var(--fg)]",
        Tag === "h2" && "mt-10 mb-3 text-2xl",
        Tag === "h3" && "mt-8 mb-2 text-xl",
        Tag === "h4" && "mt-6 mb-2 text-lg",
        className,
      )}
    >
      {children}
    </Tag>
  );
}

async function Pre(props: {
  children?: ReactNode;
  raw?: string;
  "data-language"?: string;
  "data-filename"?: string;
  "data-badge"?: string;
}) {
  const child = props.children as
    | { props?: { children?: string; className?: string } }
    | string
    | undefined;

  let code = props.raw ?? "";
  let lang = props["data-language"] ?? "text";

  if (typeof child === "object" && child?.props) {
    const className = child.props.className ?? "";
    const match = /language-([\w#+-]+)/.exec(className);
    if (match) lang = match[1]!;
    code = String(child.props.children ?? "").replace(/\n$/, "");
  } else if (typeof child === "string") {
    code = child;
  }

  const html = await highlightCode(code, lang);
  return (
    <CodeBlock
      html={html}
      code={code}
      language={lang}
      filename={props["data-filename"]}
      badge={props["data-badge"]}
    />
  );
}

export const mdxComponents = {
  h2: (props: { id?: string; children?: ReactNode }) => (
    <Heading as="h2" {...props} />
  ),
  h3: (props: { id?: string; children?: ReactNode }) => (
    <Heading as="h3" {...props} />
  ),
  h4: (props: { id?: string; children?: ReactNode }) => (
    <Heading as="h4" {...props} />
  ),
  p: (props: { children?: ReactNode }) => (
    <p className="my-3 leading-7 text-[color:var(--fg-muted)]">{props.children}</p>
  ),
  ul: (props: { children?: ReactNode }) => (
    <ul className="my-4 list-disc space-y-2 pl-5 text-[color:var(--fg-muted)]">
      {props.children}
    </ul>
  ),
  ol: (props: { children?: ReactNode }) => (
    <ol className="my-4 list-decimal space-y-2 pl-5 text-[color:var(--fg-muted)]">
      {props.children}
    </ol>
  ),
  li: (props: { children?: ReactNode }) => <li>{props.children}</li>,
  a: (props: { href?: string; children?: ReactNode }) => (
    <a
      href={props.href}
      className="font-medium text-[color:var(--accent)] underline-offset-4 hover:underline"
    >
      {props.children}
    </a>
  ),
  table: (props: { children?: ReactNode }) => (
    <div className="my-4 overflow-x-auto rounded-xl border border-[color:var(--border)]">
      <table className="w-full min-w-[32rem] border-collapse text-sm">
        {props.children}
      </table>
    </div>
  ),
  th: (props: { children?: ReactNode }) => (
    <th className="border-b border-[color:var(--border)] bg-[color:var(--surface)] px-3 py-2 text-left font-medium">
      {props.children}
    </th>
  ),
  td: (props: { children?: ReactNode }) => (
    <td className="border-b border-[color:var(--border)] px-3 py-2 text-[color:var(--fg-muted)]">
      {props.children}
    </td>
  ),
  pre: Pre,
  code: (props: { children?: ReactNode; className?: string }) => {
    if (props.className) return <code className={props.className}>{props.children}</code>;
    return (
      <code className="rounded bg-[color:var(--surface)] px-1.5 py-0.5 font-mono text-[0.85em] text-[color:var(--fg)]">
        {props.children}
      </code>
    );
  },
  Callout,
  StatusBadge,
  BenchmarkSuite,
  DataLoaderSimulator,
  SplitVisualizer,
  PipelineBuilder,
  InstallCommandGenerator,
  ShapeExplorer,
  Steps: ({ children }: { children?: ReactNode }) => (
    <ol className="my-6 space-y-4 border-l border-[color:var(--border)] pl-4">
      {children}
    </ol>
  ),
  Step: ({
    title,
    children,
  }: {
    title: string;
    children?: ReactNode;
  }) => (
    <li>
      <p className="font-medium text-[color:var(--fg)]">{title}</p>
      <div className="mt-1 text-sm text-[color:var(--fg-muted)]">{children}</div>
    </li>
  ),
  ApiSignature: ({
    name,
    signature,
    children,
  }: {
    name: string;
    signature: string;
    children?: ReactNode;
  }) => (
    <div className="my-6 rounded-xl border border-[color:var(--border)] bg-[color:var(--surface)]/60 p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <code className="font-mono text-sm text-[color:var(--accent)]">{name}</code>
        <StatusBadge status="design-preview" label="Design preview" />
      </div>
      <pre className="overflow-x-auto rounded-lg bg-[#0d1117] p-3 font-mono text-xs text-white/80">
        {signature}
      </pre>
      <div className="mt-3 text-sm text-[color:var(--fg-muted)]">{children}</div>
    </div>
  ),
};
