// Shared, keyboard-accessible intent-language selector for setup and settings.
// The array order is significant: the matcher tries languages in this order.
export function createLanguageSelect({ options, value, onChange }) {
  const names = { ...(options || {}) };
  let selected = [...new Set(Array.isArray(value) && value.length ? value : ['en'])];
  let disabled = false;
  const root = document.createElement('div');
  root.className = 'language-select';
  root.innerHTML = `<details class="language-picker"><summary><span class="language-pills"></span><span class="language-chevron" aria-hidden="true">▾</span></summary>
    <fieldset><legend>Home Assistant intent languages</legend><div class="language-options"></div></fieldset></details>
    <p class="help language-validation" role="status" aria-live="polite"></p>`;
  const picker = root.querySelector('details');
  const summary = root.querySelector('summary');
  const pills = root.querySelector('.language-pills');
  const list = root.querySelector('.language-options');
  const message = root.querySelector('.language-validation');

  function render() {
    pills.replaceChildren();
    for (const code of selected) {
      const pill = document.createElement('span');
      pill.className = 'language-pill';
      pill.textContent = names[code] || `${code} (unavailable)`;
      pills.appendChild(pill);
    }
    summary.setAttribute('aria-label', `Home Assistant intent languages: ${selected.map(code => names[code] || code).join(', ')}`);
    list.replaceChildren();
    // Keep previously configured, now-unavailable values visible and removable.
    const codes = [...Object.keys(names), ...selected.filter(code => !(code in names))];
    for (const code of codes) {
      const label = document.createElement('label');
      label.className = 'language-option';
      const input = document.createElement('input');
      input.type = 'checkbox'; input.value = code; input.checked = selected.includes(code);
      input.disabled = disabled;
      const title = document.createElement('span');
      title.textContent = names[code] || `${code} (unavailable)`;
      label.append(input, title);
      list.appendChild(label);
      input.addEventListener('change', () => {
        if (!input.checked && selected.length === 1) {
          input.checked = true;
          message.textContent = 'Select at least one language.';
          return;
        }
        message.textContent = '';
        selected = input.checked ? [...selected, code] : selected.filter(item => item !== code);
        render();
        // Keep keyboard focus on the option that was changed after rerendering.
        [...list.querySelectorAll('input')].find(node => node.value === code)?.focus();
        onChange?.([...selected]);
      });
    }
  }

  summary.addEventListener('click', event => { if (disabled) event.preventDefault(); });
  render();
  return {
    element: root,
    getValue: () => [...selected],
    setValue(value) { selected = [...new Set(value)]; message.textContent = ''; render(); },
    setDisabled(value) {
      disabled = value;
      if (disabled) picker.open = false;
      summary.setAttribute('aria-disabled', String(disabled));
      render();
    },
  };
}
