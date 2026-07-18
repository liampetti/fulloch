import {
    App,
    MetadataCache,
    Plugin,
    PluginSettingTab,
    Setting,
    TFile,
    Notice,
    normalizePath,
    WorkspaceLeaf,
} from "obsidian";

interface FullochSettings {
    host: string;
    port: number;
    token: string;
}

const DEFAULT_SETTINGS: FullochSettings = {
    host: "https://localhost",
    port: 8765,
    token: "",
};

interface VaultFile {
    path: string;
    name: string;
    tags: string[];
    links: string[];
    frontmatter: Record<string, unknown>;
}

interface FileContext {
    path: string;
    name: string;
    tags: string[];
    links: string[];
    backlinks: string[];
    frontmatter: Record<string, unknown>;
    selection: string;
}

export default class FullochPlugin extends Plugin {
    settings: FullochSettings;
    private ws: WebSocket | null = null;
    private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    private reconnectDelay = 3000;
    private lastSelectionKey = "";
    private statusBarItem: HTMLElement;
    private intentionalClose = false;

    async onload() {
        await this.loadSettings();

        this.statusBarItem = this.addStatusBarItem();
        this.setStatus("disconnected");

        this.addRibbonIcon("mic", "Fulloch", () => {
            if (this.ws?.readyState === WebSocket.OPEN) {
                new Notice("Fulloch: connected to " + this.settings.host);
            } else {
                this.intentionalClose = false;
                this.connect();
            }
        });

        this.addCommand({
            id: "send-context",
            name: "Send this note's context to Fulloch",
            checkCallback: (checking) => {
                const file = this.app.workspace.getActiveFile();
                if (file) {
                    if (!checking) this.sendFileContext(file);
                    return true;
                }
                return false;
            },
        });

        this.addCommand({
            id: "reconnect",
            name: "Reconnect to Fulloch",
            callback: () => {
                this.intentionalClose = false;
                this.ws?.close();
                this.connect();
            },
        });

        this.addSettingTab(new FullochSettingTab(this.app, this));

        // Connect once the workspace is ready so getActiveFile() works.
        this.app.workspace.onLayoutReady(() => {
            this.connect();
        });

        // Send context whenever the active file changes.
        this.registerEvent(
            this.app.workspace.on("active-leaf-change", (_leaf: WorkspaceLeaf | null) => {
                const file = this.app.workspace.getActiveFile();
                if (file) this.sendFileContext(file);
            })
        );
        // Obsidian does not expose a stable selection-change event across all
        // supported editor versions. Poll only the tiny selection string and
        // send context when it actually changes.
        this.registerInterval(window.setInterval(() => this.syncActiveSelection(), 500));

        // Update vault metadata when files are created, renamed, or deleted.
        this.registerEvent(
            this.app.vault.on("create", () => this.sendVaultMetadata())
        );
        this.registerEvent(
            this.app.vault.on("rename", () => this.sendVaultMetadata())
        );
        this.registerEvent(
            this.app.vault.on("delete", () => this.sendVaultMetadata())
        );

        // Notify Fulloch of content edits so the embedding index stays fresh
        // without waiting for the next mtime scan.
        this.registerEvent(
            this.app.vault.on("modify", (file) => {
                if (file instanceof TFile && file.extension === "md") {
                    this.sendFileChanged(file);
                }
            })
        );

        // First-run hint: if this is a fresh install, point the user at the
        // dashboard's "Connect Obsidian" card to get the auth token.
        const existing = (await this.loadData()) as { firstRunShown?: boolean } | null;
        if (!existing?.firstRunShown) {
            new Notice(
                "Fulloch: open the dashboard's 'Connect Obsidian' card to get your auth token, then paste it here."
            );
            await this.saveData({ ...this.settings, firstRunShown: true });
        }
    }

    onunload() {
        this.intentionalClose = true;
        if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
        this.ws?.close();
    }

    // -------------------------------------------------------------------------
    // WebSocket connection
    // -------------------------------------------------------------------------

    connect() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) return;

        const { host, port, token } = this.settings;
        const proto = host.startsWith("https://") ? "wss" : "ws";
        const cleanHost = host.replace(/^https?:\/\//, "");
        const url = `${proto}://${cleanHost}:${port}/ws/obsidian${token ? `?token=${encodeURIComponent(token)}` : ""}`;

        try {
            this.ws = new WebSocket(url);
        } catch {
            this.setStatus("error");
            this.scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            this.reconnectDelay = 3000;
            this.setStatus("connected");
            this.sendVaultMetadata();
            const file = this.app.workspace.getActiveFile();
            if (file) this.sendFileContext(file);
        };

        this.ws.onmessage = (event: MessageEvent) => {
            try {
                const cmd = JSON.parse(event.data as string);
                this.handleCommand(cmd);
            } catch {
                // ignore malformed messages
            }
        };

        this.ws.onclose = () => {
            this.setStatus("disconnected");
            if (!this.intentionalClose) this.scheduleReconnect();
        };

        this.ws.onerror = () => {
            this.setStatus("error");
        };
    }

    private scheduleReconnect() {
        if (this.reconnectTimer !== null) clearTimeout(this.reconnectTimer);
        this.reconnectTimer = setTimeout(() => {
            this.reconnectTimer = null;
            this.connect();
        }, this.reconnectDelay);
        // Back off up to 30 s.
        this.reconnectDelay = Math.min(this.reconnectDelay * 1.5, 30_000);
    }

    // -------------------------------------------------------------------------
    // Sending data to Fulloch
    // -------------------------------------------------------------------------

    private send(data: object) {
        if (this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }

    sendVaultMetadata() {
        const adapter = this.app.vault.adapter as { basePath?: string };
        const vaultPath = adapter.basePath ?? "";

        const files: VaultFile[] = this.app.vault.getMarkdownFiles().map((f) => {
            const cache = this.app.metadataCache.getFileCache(f);
            return {
                path: f.path,
                name: f.basename,
                tags: (cache?.tags ?? []).map((t) => t.tag),
                links: (cache?.links ?? []).map((l) => l.link),
                frontmatter: cache?.frontmatter ?? {},
            };
        });

        const dailyNotes = this.getDailyNotesConfig();

        this.send({
            type: "vault_metadata",
            vault_path: vaultPath,
            file_count: files.length,
            files,
            daily_notes: dailyNotes,
        });
    }

    sendFileContext(file: TFile) {
        const cache = this.app.metadataCache.getFileCache(file);
        const tags = (cache?.tags ?? []).map((t) => t.tag);
        const links = (cache?.links ?? []).map((l) => l.link);
        const frontmatter = cache?.frontmatter ?? {};
        const activeFile = this.app.workspace.getActiveFile();
        const selection = activeFile?.path === file.path
            ? (this.app.workspace.activeEditor?.editor?.getSelection() ?? "")
            : "";

        // Resolved backlinks: which files link TO this file.
        const resolvedLinks = (this.app.metadataCache as MetadataCache & {
            resolvedLinks: Record<string, Record<string, number>>;
        }).resolvedLinks;
        const backlinks: string[] = [];
        for (const [sourcePath, targets] of Object.entries(resolvedLinks)) {
            if (targets[file.path]) {
                backlinks.push(sourcePath.replace(/\.md$/, ""));
            }
        }

        const ctx: FileContext = {
            path: file.path,
            name: file.basename,
            tags,
            links,
            backlinks,
            frontmatter,
            selection,
        };

        this.send({ type: "context", file: ctx });
    }

    private syncActiveSelection() {
        const file = this.app.workspace.getActiveFile();
        if (!file) return;
        const selection = this.app.workspace.activeEditor?.editor?.getSelection() ?? "";
        const key = `${file.path}\0${selection}`;
        if (key === this.lastSelectionKey) return;
        this.lastSelectionKey = key;
        this.sendFileContext(file);
    }

    sendFileChanged(file: TFile) {
        this.send({ type: "file_changed", path: file.path });
    }

    // -------------------------------------------------------------------------
    // Receiving commands from Fulloch
    // -------------------------------------------------------------------------

    private handleCommand(cmd: Record<string, unknown>) {
        switch (cmd.type) {
            case "ping":
                this.send({ type: "pong" });
                break;
            case "open_file":
                this.openFile(cmd.path as string);
                break;
            case "insert":
                this.insertText(cmd.text as string, cmd.file as string | undefined);
                break;
            case "rename_active":
                this.renameActiveNote(cmd.title as string);
                break;
            case "delete_active":
                this.deleteActiveNote();
                break;
            case "replace_selection":
                this.replaceSelection(cmd.text as string);
                break;
            case "vault_rejected":
                const reason = cmd.reason as string;
                const messages: Record<string, string> = {
                    not_a_vault: "Fulloch says the reported path is not a vault (no .obsidian/ folder).",
                    unreadable: "Fulloch can't read the reported vault path.",
                    missing: "Fulloch says the vault path is missing.",
                };
                this.setStatus("error", messages[reason] || `Fulloch rejected the vault: ${reason}`);
                break;
        }
    }

    private async openFile(absolutePath: string) {
        // absolutePath may be absolute (from the server) or vault-relative.
        const adapter = this.app.vault.adapter as { basePath?: string };
        const vaultBase = adapter.basePath ?? "";
        let vaultRelative = absolutePath;
        if (vaultBase && absolutePath.startsWith(vaultBase)) {
            vaultRelative = absolutePath.slice(vaultBase.length).replace(/^[/\\]/, "");
        }
        const file = this.app.vault.getAbstractFileByPath(vaultRelative);
        if (file instanceof TFile) {
            const leaf = this.app.workspace.getLeaf(false);
            await leaf.openFile(file);
        }
    }

    private async insertText(text: string, targetFile?: string) {
        if (targetFile) {
            // Append to a specific file by vault-relative path.
            const file = this.app.vault.getAbstractFileByPath(targetFile);
            if (file instanceof TFile) {
                const existing = await this.app.vault.read(file);
                await this.app.vault.modify(file, existing + "\n" + text);
            }
            return;
        }
        // Insert at cursor in the active editor.
        const editor = this.app.workspace.activeEditor?.editor;
        if (editor) {
            const cursor = editor.getCursor("to");
            editor.replaceRange("\n" + text, cursor);
            new Notice("Fulloch: inserted text");
        }
    }

    private async renameActiveNote(title: string) {
        const file = this.app.workspace.getActiveFile();
        const cleanTitle = title.trim().replace(/[\\/:*?"<>|]/g, "-");
        if (!file || !cleanTitle) {
            new Notice("Fulloch: open a note before renaming it");
            return;
        }
        const parent = file.parent?.path;
        const targetPath = normalizePath(`${parent ? parent + "/" : ""}${cleanTitle}.md`);
        if (targetPath === file.path) return;
        try {
            await this.app.vault.rename(file, targetPath);
            this.sendFileContext(file);
            new Notice(`Fulloch: renamed note to ${cleanTitle}`);
        } catch {
            new Notice("Fulloch: couldn't rename the active note");
        }
    }

    private async deleteActiveNote() {
        const file = this.app.workspace.getActiveFile();
        if (!file) {
            new Notice("Fulloch: open a note before deleting it");
            return;
        }
        try {
            await this.app.vault.trash(file, true);
            new Notice("Fulloch: moved the active note to Obsidian trash");
        } catch {
            new Notice("Fulloch: couldn't delete the active note");
        }
    }

    private replaceSelection(text: string) {
        const editor = this.app.workspace.activeEditor?.editor;
        if (!editor || !editor.getSelection()) {
            new Notice("Fulloch: select text before asking me to replace it");
            return;
        }
        editor.replaceSelection(text);
        new Notice("Fulloch: replaced selected text");
    }

    // -------------------------------------------------------------------------
    // Daily notes config detection
    // -------------------------------------------------------------------------

    private getDailyNotesConfig(): Record<string, string> | null {
        try {
            const internal = (this.app as App & {
                internalPlugins?: {
                    plugins?: Record<string, {
                        enabled?: boolean;
                        instance?: { options?: Record<string, string> };
                    }>;
                };
            }).internalPlugins;
            const dn = internal?.plugins?.["daily-notes"];
            if (dn?.enabled) {
                const opts = dn.instance?.options ?? {};
                return {
                    folder: opts["folder"] ?? "",
                    format: opts["format"] ?? "YYYY-MM-DD",
                    template: opts["template"] ?? "",
                };
            }
        } catch {
            // plugin not available
        }
        return null;
    }

    // -------------------------------------------------------------------------
    // Status bar
    // -------------------------------------------------------------------------

    private setStatus(state: "connected" | "disconnected" | "error", detail?: string) {
        const icons = { connected: "◉", disconnected: "○", error: "⚠" };
        const tooltip =
            state === "connected"
                ? `Fulloch: connected to ${this.settings.host}:${this.settings.port}`
                : state === "error"
                ? detail || "Fulloch: connection error — will retry"
                : "Fulloch: disconnected";
        this.statusBarItem.setText(`Fulloch ${icons[state]}`);
        this.statusBarItem.title = tooltip;
    }

    // -------------------------------------------------------------------------
    // Settings persistence
    // -------------------------------------------------------------------------

    async loadSettings() {
        this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
    }

    async saveSettings() {
        await this.saveData(this.settings);
    }
}

// -----------------------------------------------------------------------------
// Settings tab
// -----------------------------------------------------------------------------

class FullochSettingTab extends PluginSettingTab {
    plugin: FullochPlugin;

    constructor(app: App, plugin: FullochPlugin) {
        super(app, plugin);
        this.plugin = plugin;
    }

    display(): void {
        const { containerEl } = this;
        containerEl.empty();

        containerEl.createEl("p", {
            text: "Connect this vault to your Fulloch voice assistant. Fulloch will know which note you have open and navigate to notes it writes.",
            cls: "setting-item-description",
        });

        new Setting(containerEl)
            .setName("Fulloch host")
            .setDesc("HTTPS URL of your Fulloch instance (for example, https://localhost)")
            .addText((text) =>
                text
                    .setPlaceholder("https://localhost")
                    .setValue(this.plugin.settings.host)
                    .onChange(async (value) => {
                        this.plugin.settings.host = value.trim();
                        await this.plugin.saveSettings();
                    })
            );

        new Setting(containerEl)
            .setName("Dashboard port")
            .setDesc("Port Fulloch is running on (default 8765)")
            .addText((text) =>
                text
                    .setPlaceholder("8765")
                    .setValue(String(this.plugin.settings.port))
                    .onChange(async (value) => {
                        this.plugin.settings.port = parseInt(value) || 8765;
                        await this.plugin.saveSettings();
                    })
            );

        new Setting(containerEl)
            .setName("Auth token")
            .setDesc("Auth token from the dashboard's Obsidian tab (leave blank if not set)")
            .addText((text) =>
                text
                    .setPlaceholder("optional")
                    .setValue(this.plugin.settings.token)
                    .onChange(async (value) => {
                        this.plugin.settings.token = value.trim();
                        await this.plugin.saveSettings();
                    })
            );

        new Setting(containerEl)
            .setName("Connection")
            .setDesc("Reconnect to Fulloch with the current settings")
            .addButton((btn) =>
                btn.setButtonText("Reconnect").onClick(() => {
                    this.plugin.ws?.close();
                    (this.plugin as FullochPlugin & { intentionalClose: boolean }).intentionalClose = false;
                    this.plugin.connect();
                    new Notice("Fulloch: reconnecting…");
                })
            );
    }
}
