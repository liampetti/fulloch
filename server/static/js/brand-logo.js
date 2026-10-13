import { createLifetime } from './browser-lifetime.js';
import { createLogoSprites } from './brand-logo-sprites.js';

export function createBrandLogo(canvas) {
  const lifetime = createLifetime();
  const context = canvas.getContext('2d');
  const sheet = new Image();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let loaded = false;
  let sprites = null;
  let talking = false;
  let frame = 0;
  let timer = null;

  const randomDelay = (min, max) => min + Math.random() * (max - min);
  const showFrame = index => {
    frame = index;
    if (!loaded || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(sprites, index % 4 * 314, Math.floor(index / 4) * 314,
      314, 314, 0, 0, canvas.width, canvas.height);
  };
  const schedule = (callback, delay) => {
    timer = lifetime.setTimeout(() => { timer = null; callback(); }, delay);
  };
  const blink = (step = 0) => {
    showFrame([1, 2, 1, 0][step]); // F02 → F03 → F02 → F01
    if (step === 3) scheduleIdle();
    else schedule(() => blink(step + 1), step === 1 ? 110 : 80);
  };
  const scheduleIdle = () => {
    schedule(() => {
      if (Math.random() < 0.25) {
        showFrame(4 + Math.floor(Math.random() * 4)); // F05–F08
        schedule(() => { showFrame(0); scheduleIdle(); }, randomDelay(650, 1400));
      } else {
        blink();
      }
    }, randomDelay(2500, 6000));
  };
  const animateTalking = () => {
    let next = 8 + Math.floor(Math.random() * 8); // F09–F16
    if (next === frame) next = 8 + (next - 8 + 1) % 8;
    showFrame(next);
    schedule(animateTalking, randomDelay(90, 180));
  };
  const syncAnimation = () => {
    if (timer !== null) { lifetime.clearTimeout(timer); timer = null; }
    showFrame(0);
    if (!loaded || !context || lifetime.destroyed || reducedMotion.matches || document.hidden) return;
    if (talking) animateTalking();
    else scheduleIdle();
  };

  lifetime.listen(sheet, 'load', () => {
    sprites = createLogoSprites(sheet);
    loaded = true;
    syncAnimation();
  });
  lifetime.listen(document, 'visibilitychange', syncAnimation);
  lifetime.listen(reducedMotion, 'change', syncAnimation);
  sheet.src = '/fulloch_animate.png';

  return {
    setTalking(value) {
      if (lifetime.destroyed || talking === !!value) return;
      talking = !!value;
      syncAnimation();
    },
    destroy() { lifetime.destroy(); showFrame(0); },
  };
}
