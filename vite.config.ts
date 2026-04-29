import rawPlugin from 'vite-raw-plugin';

export default {
  plugins: [
    rawPlugin({
      fileRegex: /\.wgsl$/,
    }),
  ],
  base: '/3dgs/'
};