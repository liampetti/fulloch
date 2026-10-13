const CELL_SIZE = 314;
const PADDING = 2;

// Source crops from utils/fulloch_animate.md. The artwork moves inside these
// cells, so only its facial features are transferred onto the fixed F01 body.
const FRAMES = [
  [0, 0, 314, 314], [314, 0, 313, 314],
  [627, 0, 314, 314], [941, 0, 313, 314],
  [0, 314, 314, 313], [314, 314, 313, 313],
  [627, 314, 314, 313], [941, 314, 313, 313],
  [0, 627, 314, 314], [314, 627, 313, 314],
  [627, 627, 314, 314], [941, 627, 313, 314],
  [0, 941, 314, 313], [314, 941, 313, 313],
  [627, 941, 314, 313], [941, 941, 313, 313],
];
const EYES = {
  0: [[130, 101, 27, 34], [225, 101, 27, 35]],
  1: [[114, 117, 29, 18], [213, 117, 29, 18]],
  2: [[97, 110, 36, 18], [197, 110, 36, 18]],
  4: [[95, 91, 26, 35], [189, 91, 29, 37]],
  5: [[119, 90, 28, 35], [216, 90, 26, 35]],
  6: [[97, 86, 30, 29], [198, 88, 30, 34]],
  7: [[99, 106, 28, 32], [197, 105, 26, 33]],
};
const MOUTHS = {
  0: [175, 138, 35, 15],
  8: [171, 121, 37, 29], 9: [161, 120, 42, 35],
  10: [137, 115, 51, 43], 11: [149, 123, 23, 30],
  12: [135, 115, 39, 31], 13: [160, 114, 44, 34],
  14: [138, 112, 50, 38], 15: [144, 126, 40, 31],
};

const createCanvas = (width, height) => {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  return canvas;
};
const padded = ([x, y, width, height]) =>
  [x - PADDING, y - PADDING, width + PADDING * 2, height + PADDING * 2];
const featureOpacity = (data, i, eye) => {
  const [r, g, b] = data.subarray(i, i + 3);
  // Remove green skin, retaining ink, eye highlights and pink mouth interiors.
  // Cheeks are deliberately excluded from eye layers so they cannot move.
  if (eye && r > g + 12 && r > b + 30) return 0;
  return Math.max(0, Math.min(1, (30 - Math.min(g - r, g - b)) / 24));
};

function extractFeature(sheet, index, bounds, eye) {
  const [x, y, width, height] = padded(bounds);
  const canvas = createCanvas(width, height);
  const context = canvas.getContext('2d');
  context.drawImage(sheet, FRAMES[index][0] + x, FRAMES[index][1] + y,
    width, height, 0, 0, width, height);
  const pixels = context.getImageData(0, 0, width, height);
  for (let i = 0; i < pixels.data.length; i += 4) {
    pixels.data[i + 3] *= featureOpacity(pixels.data, i, eye);
  }
  context.putImageData(pixels, 0, 0);
  return canvas;
}

function eraseFeature(pixels, bounds, eye) {
  const [x, y, width, height] = padded(bounds);
  const { data } = pixels;
  const skinAt = (cx, cy) => {
    const sum = [0, 0, 0];
    let count = 0;
    for (let sy = cy - 8; sy <= cy + 8; sy++) {
      for (let sx = cx - 8; sx <= cx + 8; sx++) {
        const i = (sy * CELL_SIZE + sx) * 4;
        if (Math.min(data[i + 1] - data[i], data[i + 1] - data[i + 2]) < 30) continue;
        for (let channel = 0; channel < 3; channel++) sum[channel] += data[i + channel];
        count++;
      }
    }
    const i = (cy * CELL_SIZE + cx) * 4;
    return count ? sum.map(value => value / count) : Array.from(data.subarray(i, i + 3));
  };
  const corners = [skinAt(x, y), skinAt(x + width, y),
    skinAt(x, y + height), skinAt(x + width, y + height)];
  // Reconstruct the small patches of skin hidden by the idle eyes and smile.
  // All other pixels (including cheeks, outline and headset) remain untouched.
  for (let sy = 0; sy < height; sy++) {
    for (let sx = 0; sx < width; sx++) {
      const i = ((y + sy) * CELL_SIZE + x + sx) * 4;
      if (eye && data[i] > data[i + 1] + 12 && data[i] > data[i + 2] + 30) continue;
      // Fill the whole small skin patch, including faint anti-aliased edges,
      // so closed eyes and narrow mouths never leave an idle-feature outline.
      const u = sx / (width - 1), v = sy / (height - 1);
      for (let channel = 0; channel < 3; channel++) {
        data[i + channel] = (1 - v) * ((1 - u) * corners[0][channel] + u * corners[1][channel])
          + v * ((1 - u) * corners[2][channel] + u * corners[3][channel]);
      }
    }
  }
}

export function createLogoSprites(sheet) {
  const body = createCanvas(CELL_SIZE, CELL_SIZE);
  const context = body.getContext('2d');
  context.drawImage(sheet, ...FRAMES[0], 0, 0, CELL_SIZE, CELL_SIZE);
  const pixels = context.getImageData(0, 0, CELL_SIZE, CELL_SIZE);
  EYES[0].forEach(bounds => eraseFeature(pixels, bounds, true));
  eraseFeature(pixels, MOUTHS[0], false);
  context.putImageData(pixels, 0, 0);

  const eyes = Object.fromEntries(Object.entries(EYES).map(([index, bounds]) =>
    [index, bounds.map(box => extractFeature(sheet, index, box, true))]));
  const mouths = Object.fromEntries(Object.entries(MOUTHS).map(([index, bounds]) =>
    [index, extractFeature(sheet, index, bounds, false)]));
  const sprites = createCanvas(CELL_SIZE * 4, CELL_SIZE * 4);
  const output = sprites.getContext('2d');
  const drawFeature = (feature, x, y) => output.drawImage(feature,
    Math.round(x - feature.width / 2), Math.round(y - feature.height / 2));
  for (let index = 0; index < FRAMES.length; index++) {
    output.save();
    output.translate(index % 4 * CELL_SIZE, Math.floor(index / 4) * CELL_SIZE);
    output.drawImage(body, 0, 0);
    // Ordinary talking frames keep the idle eyes; F15 borrows the upward glance.
    const eyeIndex = index === 14 ? 6 : EYES[index] ? index : 0;
    const dx = eyeIndex === 4 ? -4 : eyeIndex === 5 ? 4 : 0;
    const dy = eyeIndex === 1 ? 8 : eyeIndex === 6 ? -3 : eyeIndex === 7 ? 3 : 0;
    drawFeature(eyes[eyeIndex][0], 143.5 + dx, 118 + dy);
    drawFeature(eyes[eyeIndex][1], 238.5 + dx, 118.5 + dy);
    drawFeature(mouths[index] || mouths[0], 192.5, 145.5);
    output.restore();
  }
  return sprites;
}
