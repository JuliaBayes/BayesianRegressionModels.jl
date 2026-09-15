import { defineConfig } from 'vitepress'
import { tabsMarkdownPlugin } from 'vitepress-plugin-tabs'
import footnote from "markdown-it-footnote";
import path from 'path'
import { stanLanguage } from './stan-language'

function getBaseRepository(base: string): string {
  if (!base || base === '/') return '/';
  const parts = base.split('/').filter(Boolean);
  return parts.length > 0 ? `/${parts[0]}/` : '/';
}

const baseTemp = {
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
}

const navTemp = {
  nav: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
}

const nav = [
  ...navTemp.nav,
  {
    component: 'VersionPicker'
  }
]

// https://vitepress.dev/reference/site-config
export default defineConfig({
  base: 'REPLACE_ME_DOCUMENTER_VITEPRESS',// TODO: replace this in makedocs!
  title: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  description: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
  lastUpdated: true,
  cleanUrls: true,
  // Allow `http://localhost:…` references (gallery.md links to the
  // local BRM web app) — vitepress build flags them as dead links
  // otherwise. Also tolerate Documenter's `./@ref` artifact: when a
  // `[`x`](@ref)` in a docstring can't be resolved and make.jl runs
  // with `warnonly=true`, Documenter emits the link as a literal
  // `./@ref`, which vitepress then flags as a dead link.
  ignoreDeadLinks: [/^https?:\/\/localhost(:\d+)?(\/|$)/, /@ref/],
  outDir: 'REPLACE_ME_DOCUMENTER_VITEPRESS', // This is required for MarkdownVitepress to work correctly...
  head: [
    ['link', { rel: 'icon', href: 'REPLACE_ME_DOCUMENTER_VITEPRESS_FAVICON' }],
    ['script', {src: `${getBaseRepository(baseTemp.base)}versions.js`}],
    ['script', {src: `${baseTemp.base}siteinfo.js`}],
    // HTMX runtime — for inlining live BRM gallery fragments via
    // `<div hx-get="…" hx-trigger="load">` placeholders. HTMX requests
    // carry HX-Request: true automatically, so BRM's routes return the
    // bare body fragment that drops directly into the docs page.
    ['script', {src: 'https://cdn.jsdelivr.net/npm/htmx.org@2.0.8/dist/htmx.min.js'}],
    // Vega + Vega-Lite for the embedded PPC plots' Vega-Embed runtime.
    // BRM's PPC plots are emitted as Vega-Lite specs that the page-
    // included `vega_head()` script auto-embeds; loading vega here
    // makes those specs render inside the docs page too.
    ['script', {src: 'https://cdn.jsdelivr.net/npm/vega@5'}],
    ['script', {src: 'https://cdn.jsdelivr.net/npm/vega-lite@5'}],
    ['script', {src: 'https://cdn.jsdelivr.net/npm/vega-embed@6'}],
    // Map HTMXObjects' --htmxo-* theme variables to VitePress's
    // brand/state tokens so embedded gallery components match the
    // docs theme automatically. HTMXO defaults remain as fallback.
    ['style', {}, `
:root {
    --htmxo-accent:  var(--vp-c-brand-1, #4a90d9);
    --htmxo-success: var(--vp-c-success-1, #2a9d8f);
    --htmxo-warning: var(--vp-c-warning-1, #e9a23b);
    --htmxo-error:   var(--vp-c-danger-1, #e76f51);
    --htmxo-border:  var(--vp-c-divider, currentColor);
    --htmxo-muted:   var(--vp-c-text-3, color-mix(in srgb, currentColor 60%, transparent));
}
    `]
  ],

  markdown: {
    math: true,
    config(md) {
      md.use(tabsMarkdownPlugin);
      md.use(footnote);
    },
    theme: {
      light: "github-light",
      dark: "github-dark"
    },
    languages: [stanLanguage],
  },
  vite: {
    define: {
      __DEPLOY_ABSPATH__: JSON.stringify('REPLACE_ME_DOCUMENTER_VITEPRESS_DEPLOY_ABSPATH'),
    },
    server: {
      // Bind to all interfaces so the dev server is reachable from
      // other devices on the local network.
      host: true,
      proxy: {
        // Live BRM gallery embedding (dev only). `<div
        // hx-get="/live-brm/…">` forwards to the running BRM web app
        // on :8121 so the docs page shows live state. In production,
        // point the same path at recordings produced by `record!` —
        // same markdown source. Override the target via
        // `BRM_DEV_TARGET=http://host:port` env.
        '/live-brm': {
          target: process.env.BRM_DEV_TARGET || 'http://localhost:8121',
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/live-brm/, ''),
        }
      }
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '../components')
      }
    },
  },
  themeConfig: {
    outline: 'deep',
    logo: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    search: {
      provider: 'local',
      options: {
        detailedView: true
      }
    },
    nav,
    sidebar: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    editLink: 'REPLACE_ME_DOCUMENTER_VITEPRESS',
    socialLinks: [
      { icon: 'github', link: 'REPLACE_ME_DOCUMENTER_VITEPRESS' }
    ],
    footer: {
      message: 'Made with <a href="https://luxdl.github.io/DocumenterVitepress.jl/dev/" target="_blank"><strong>DocumenterVitepress.jl</strong></a><br>',
      copyright: `© Copyright ${new Date().getUTCFullYear()}.`
    }
  }
})
