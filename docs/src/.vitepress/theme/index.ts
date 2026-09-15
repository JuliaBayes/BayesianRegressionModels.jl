// .vitepress/theme/index.ts
import DefaultTheme from 'vitepress/theme'
import type { Theme as ThemeConfig } from 'vitepress'
import { onMounted } from 'vue'

import { enhanceAppWithTabs } from 'vitepress-plugin-tabs/client'

// Synced from HTMXObjects/assets/vitepress/htmxo-embed.ts by
// `HTMXObjects.vitepress_theme_install` in make.jl. Don't edit in place
// — edit the upstream and re-run make.jl.
import { setupHtmxoEmbed } from './htmxo-embed'
import { setupBackendComparisons } from './backend-comparison'

import './style.css'

export const Theme: ThemeConfig = {
  extends: DefaultTheme,
  setup() {
    // Tab enhancement moves SSR code blocks; wait until Vue has hydrated them.
    onMounted(setupBackendComparisons)
  },
  enhanceApp({ app, router }) {
    enhanceAppWithTabs(app);
    // HTMXObjects embed wiring: data-hx-base resolution + SPA route
    // re-process + .htmxo-embed link rewriting. BRM defaults the proxy
    // prefix to `/live-brm` (matches the Vite proxy in config.mts and
    // committed recordings under public/live-brm/).
    // BRM gallery cards embed inline `<script>AoV.embed(…)>` calls
    // for the auto-PPC plot. Load vega + vega-embed + BRM's AoV
    // runtime route into the docs page so those scripts succeed when
    // htmx evaluates them on swap.
    setupHtmxoEmbed(router, {
      proxyPrefix: '/live-brm',
      extraScripts: [
        'https://cdn.jsdelivr.net/npm/vega@5',
        'https://cdn.jsdelivr.net/npm/vega-lite@5',
        'https://cdn.jsdelivr.net/npm/vega-embed@6',
        '/live-brm/pipeline/aov_runtime_js',
      ],
    });
  }
}
export default Theme
