const action = document.getElementById('document-action');
const editor = document.getElementById('document-editor');
const content = document.getElementById('document-content');
const error = document.getElementById('document-error');
const pen = '<path d="m16 3 5 5L8 21H3v-5ZM14 5l5 5"/>';
const save = '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h12l4 4v12a2 2 0 0 1-2 2Z"/><path d="M7 3v6h10V3M7 21v-8h10v8"/>';
let editing = false;

const setEditing = (value) => {
  editing = value;
  editor.hidden = !value;
  content.hidden = value;
  const label = value ? 'Save document' : 'Edit document';
  action.setAttribute('aria-label', label);
  action.title = label;
  action.querySelector('svg').innerHTML = value ? save : pen;
  if (value) editor.focus();
};

action.addEventListener('click', async () => {
  error.hidden = true;
  if (!editing) {
    setEditing(true);
    return;
  }
  action.disabled = true;
  editor.readOnly = true;
  try {
    const response = await fetch(location.pathname, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: editor.value }),
    });
    if (!response.ok) throw new Error(response.status === 401
      ? 'Your session expired. Sign in to Fulloch in another tab, then try saving again.'
      : 'Could not save. Your edits are still here; please try again.');
    const result = await response.json();
    content.innerHTML = result.html;
    setEditing(false);
  } catch (e) {
    error.textContent = e.message || 'Could not save. Please try again.';
    error.hidden = false;
  } finally {
    action.disabled = false;
    editor.readOnly = false;
  }
});
