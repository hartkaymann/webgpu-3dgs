import rawPlugin from 'vite-raw-plugin';
import react from '@vitejs/plugin-react';

export default {
  plugins: [
    react(),
    rawPlugin({
      fileRegex: /\.wgsl$/,
    }),
  ],
  base: '/3dgs/'
};
