// Each component owns its listeners and scheduled work. Disposing one component
// cannot stop another component's timers or remove its event handlers.
export function createLifetime() {
  const controller = new AbortController();
  const timeouts = new Set();
  const intervals = new Set();
  const clearTimers = () => {
    timeouts.forEach(id => window.clearTimeout(id));
    intervals.forEach(id => window.clearInterval(id));
    timeouts.clear();
    intervals.clear();
  };
  return {
    get destroyed() { return controller.signal.aborted; },
    get signal() { return controller.signal; },
    listen(target, event, callback, options = {}) {
      target.addEventListener(event, callback, { ...options, signal: controller.signal });
    },
    setTimeout(callback, delay) {
      if (controller.signal.aborted) return null;
      const id = window.setTimeout(() => {
        timeouts.delete(id);
        callback();
      }, delay);
      timeouts.add(id);
      return id;
    },
    clearTimeout(id) { window.clearTimeout(id); timeouts.delete(id); },
    setInterval(callback, delay) {
      if (controller.signal.aborted) return null;
      const id = window.setInterval(callback, delay);
      intervals.add(id);
      return id;
    },
    clearInterval(id) { window.clearInterval(id); intervals.delete(id); },
    clearTimers,
    destroy() { controller.abort(); clearTimers(); },
  };
}
