import { defineConfig } from 'vitest/config'
import path from 'path'

export default defineConfig({
  resolve: {
    alias: {
      '@forgesyte/ui-core': path.resolve(__dirname, './__mocks__/ui-core.js'),
      '../../../web-ui/src/plugin-system/uiPluginManager': path.resolve(__dirname, './__mocks__/uiPluginManager.js')
    },
  },
  esbuild: {
    jsx: 'automatic',
  },
  test: {
    globals: true,
    environment: 'jsdom',
  },
})
