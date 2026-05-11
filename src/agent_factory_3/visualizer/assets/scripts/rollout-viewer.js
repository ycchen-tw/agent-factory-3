/**
 * Rollout Viewer Interactive Logic
 *
 * Handles resizable sidebar, group toggling, and rollout selection.
 */

/* ==========================================================================
   Resizable Sidebar
   ========================================================================== */

class ResizableSidebar {
  /**
   * Create a resizable sidebar.
   * @param {HTMLElement} sidebarEl - The sidebar element
   * @param {HTMLElement} handleEl - The resize handle element
   * @param {Object} options - Configuration options
   */
  constructor(sidebarEl, handleEl, options = {}) {
    this.sidebar = sidebarEl;
    this.handle = handleEl;
    this.options = {
      storageKey: 'rollout-viewer-sidebar-width',
      minWidth: 240,
      maxWidth: 500,
      defaultWidth: 320,
      ...options
    };

    this.isResizing = false;
    this.startX = 0;
    this.startWidth = 0;

    this.init();
  }

  init() {
    // Restore saved width from localStorage
    this.restoreWidth();

    // Bind event handlers
    this.handleMouseDown = this.handleMouseDown.bind(this);
    this.handleMouseMove = this.handleMouseMove.bind(this);
    this.handleMouseUp = this.handleMouseUp.bind(this);
    this.handleDoubleClick = this.handleDoubleClick.bind(this);

    // Attach listeners
    this.handle.addEventListener('mousedown', this.handleMouseDown);
    document.addEventListener('mousemove', this.handleMouseMove);
    document.addEventListener('mouseup', this.handleMouseUp);
    this.handle.addEventListener('dblclick', this.handleDoubleClick);
  }

  restoreWidth() {
    try {
      const savedWidth = localStorage.getItem(this.options.storageKey);
      if (savedWidth) {
        const width = parseInt(savedWidth, 10);
        if (width >= this.options.minWidth && width <= this.options.maxWidth) {
          this.sidebar.style.width = `${width}px`;
        }
      }
    } catch (e) {
      // localStorage not available in sandboxed iframe (e.g., wandb)
    }
  }

  saveWidth() {
    try {
      localStorage.setItem(this.options.storageKey, this.sidebar.offsetWidth.toString());
    } catch (e) {
      // localStorage not available in sandboxed iframe
    }
  }

  handleMouseDown(e) {
    this.isResizing = true;
    this.startX = e.clientX;
    this.startWidth = this.sidebar.offsetWidth;

    document.body.classList.add('resizing');
    this.handle.classList.add('resizing');

    e.preventDefault();
  }

  handleMouseMove(e) {
    if (!this.isResizing) return;

    const diff = e.clientX - this.startX;
    const newWidth = Math.max(
      this.options.minWidth,
      Math.min(this.startWidth + diff, this.options.maxWidth)
    );

    this.sidebar.style.width = `${newWidth}px`;
  }

  handleMouseUp() {
    if (!this.isResizing) return;

    this.isResizing = false;
    document.body.classList.remove('resizing');
    this.handle.classList.remove('resizing');

    this.saveWidth();
  }

  handleDoubleClick() {
    // Reset to default width
    this.sidebar.style.width = `${this.options.defaultWidth}px`;
    try {
      localStorage.removeItem(this.options.storageKey);
    } catch (e) {
      // localStorage not available in sandboxed iframe
    }
  }

  destroy() {
    this.handle.removeEventListener('mousedown', this.handleMouseDown);
    document.removeEventListener('mousemove', this.handleMouseMove);
    document.removeEventListener('mouseup', this.handleMouseUp);
    this.handle.removeEventListener('dblclick', this.handleDoubleClick);
  }
}

/* ==========================================================================
   Rollout Viewer Application
   ========================================================================== */

class RolloutViewerApp {
  /**
   * Create the Rollout Viewer application.
   * @param {Object} conversationsData - Map of rollout_id to conversation HTML
   */
  constructor(conversationsData) {
    this.conversationsData = conversationsData;
    this.selectedRolloutId = null;
    this.resizableSidebar = null;

    this.init();
  }

  init() {
    // Initialize resizable sidebar
    const sidebar = document.getElementById('sidebar');
    const resizeHandle = document.getElementById('resizeHandle');

    if (sidebar && resizeHandle) {
      this.resizableSidebar = new ResizableSidebar(sidebar, resizeHandle);
    }

    // Bind event handlers
    this.bindGroupHeaders();
    this.bindRolloutItems();

    // Select first rollout if available
    this.selectFirstRollout();
  }

  bindGroupHeaders() {
    const headers = document.querySelectorAll('.group-header');
    headers.forEach(header => {
      header.addEventListener('click', () => {
        const card = header.closest('.group-card');
        if (card) {
          card.classList.toggle('expanded');
        }
      });
    });
  }

  bindRolloutItems() {
    const items = document.querySelectorAll('.rollout-item');
    items.forEach(item => {
      item.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent group toggle
        const rolloutId = item.dataset.rolloutId;
        if (rolloutId) {
          this.selectRollout(rolloutId);
        }
      });
    });
  }

  selectFirstRollout() {
    const firstItem = document.querySelector('.rollout-item');
    if (firstItem) {
      const rolloutId = firstItem.dataset.rolloutId;
      if (rolloutId) {
        // Expand the parent group
        const parentCard = firstItem.closest('.group-card');
        if (parentCard) {
          parentCard.classList.add('expanded');
        }
        this.selectRollout(rolloutId);
      }
    }
  }

  selectRollout(rolloutId) {
    // Update selection state
    this.selectedRolloutId = rolloutId;

    // Update UI - remove previous selection
    const previousSelected = document.querySelector('.rollout-item.selected');
    if (previousSelected) {
      previousSelected.classList.remove('selected');
    }

    // Add selection to new item
    const newSelected = document.querySelector(`.rollout-item[data-rollout-id="${rolloutId}"]`);
    if (newSelected) {
      newSelected.classList.add('selected');
    }

    // Update main content
    this.displayConversation(rolloutId);
  }

  displayConversation(rolloutId) {
    const container = document.getElementById('conversationContainer');
    if (!container) return;

    const data = this.conversationsData[rolloutId];

    if (data && data.html) {
      container.innerHTML = `<div class="conversation">${data.html}</div>`;
      // Scroll to top
      container.scrollTop = 0;
    } else if (data && data.error) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">&#9888;</div>
          <div class="text-error">Error: ${data.error}</div>
        </div>
      `;
    } else {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">&#128196;</div>
          <div>No conversation data available</div>
        </div>
      `;
    }

    // Update header info
    this.updateHeaderInfo(rolloutId);

    // Update info panels (config, error)
    this.updateInfoPanels(rolloutId);
  }

  updateInfoPanels(rolloutId) {
    const panelsContainer = document.getElementById('infoPanels');
    if (!panelsContainer) return;

    const data = this.conversationsData[rolloutId];
    if (!data) {
      panelsContainer.innerHTML = '';
      return;
    }

    let panelsHtml = '';

    // Error panel (only for failed rollouts with traceback)
    if (!data.success && data.traceback) {
      const errorMsg = data.error || 'Unknown error';
      panelsHtml += `
        <div class="error-panel">
          <div class="error-panel-header">
            <span class="error-icon">&#10007;</span>
            <span>${this.escapeHtml(errorMsg)}</span>
          </div>
          <details>
            <summary>Full Traceback</summary>
            <div class="traceback">${this.escapeHtml(data.traceback)}</div>
          </details>
        </div>
      `;
    }

    // Config panel
    if (data.config_snapshot && Object.keys(data.config_snapshot).length > 0) {
      const configItems = Object.entries(data.config_snapshot)
        .map(([key, value]) => `
          <div class="config-item">
            <span class="config-item-key">${this.escapeHtml(key)}</span>
            <span class="config-item-value">${this.escapeHtml(String(value))}</span>
          </div>
        `).join('');

      panelsHtml += `
        <details class="config-panel">
          <summary>Config</summary>
          <div class="config-panel-content">
            ${configItems}
          </div>
        </details>
      `;
    }

    panelsContainer.innerHTML = panelsHtml;
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  updateHeaderInfo(rolloutId) {
    const header = document.getElementById('mainHeader');
    if (!header) return;

    // Find the rollout item to get stats
    const item = document.querySelector(`.rollout-item[data-rollout-id="${rolloutId}"]`);
    if (!item) return;

    const stats = item.dataset;
    const data = this.conversationsData[rolloutId];
    const titleEl = header.querySelector('h2');
    const infoEl = header.querySelector('.rollout-info');

    if (titleEl) {
      const statusIcon = stats.success === 'true' ? '&#10003;' : '&#10007;';
      const statusClass = stats.success === 'true' ? 'text-success' : 'text-error';
      titleEl.innerHTML = `<span class="${statusClass}">${statusIcon}</span> ${rolloutId}`;
    }

    if (infoEl) {
      let infoHtml = '';

      if (stats.reward) {
        infoHtml += `<span>Reward: <b>${parseFloat(stats.reward).toFixed(4)}</b></span>`;
      }
      if (stats.advantage) {
        const adv = parseFloat(stats.advantage);
        const advClass = adv >= 0 ? 'text-success' : 'text-error';
        infoHtml += `<span>Advantage: <b class="${advClass}">${adv >= 0 ? '+' : ''}${adv.toFixed(4)}</b></span>`;
      }
      if (stats.rounds) {
        infoHtml += `<span>Rounds: <b>${stats.rounds}</b></span>`;
      }
      if (stats.tokens) {
        infoHtml += `<span>Tokens: <b>${parseInt(stats.tokens).toLocaleString()}</b></span>`;
      }
      if (stats.time) {
        infoHtml += `<span>Time: <b>${parseFloat(stats.time).toFixed(2)}s</b></span>`;
      }
      if (data && data.weight_versions && data.weight_versions.length > 0) {
        const versions = data.weight_versions.join(' → ');
        const multiVersion = data.weight_versions.length > 1;
        const cls = multiVersion ? 'text-warning' : '';
        infoHtml += `<span>Weights: <b class="${cls}">${this.escapeHtml(versions)}</b></span>`;
      }

      infoEl.innerHTML = infoHtml;
    }
  }
}

/* ==========================================================================
   Keyboard Navigation
   ========================================================================== */

function setupKeyboardNavigation(app) {
  document.addEventListener('keydown', (e) => {
    // Only handle if not in an input field
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    const items = Array.from(document.querySelectorAll('.rollout-item'));
    if (items.length === 0) return;

    const currentIndex = items.findIndex(item => item.classList.contains('selected'));

    switch (e.key) {
      case 'ArrowDown':
      case 'j':
        e.preventDefault();
        if (currentIndex < items.length - 1) {
          const nextId = items[currentIndex + 1].dataset.rolloutId;
          if (nextId) app.selectRollout(nextId);
        }
        break;

      case 'ArrowUp':
      case 'k':
        e.preventDefault();
        if (currentIndex > 0) {
          const prevId = items[currentIndex - 1].dataset.rolloutId;
          if (prevId) app.selectRollout(prevId);
        }
        break;

      case 'Enter':
        e.preventDefault();
        // Expand/collapse current group
        const selectedItem = document.querySelector('.rollout-item.selected');
        if (selectedItem) {
          const card = selectedItem.closest('.group-card');
          if (card) card.classList.toggle('expanded');
        }
        break;
    }
  });
}

/* ==========================================================================
   Initialization
   ========================================================================== */

function initApp() {
  // conversationsData should be defined in the HTML before this script
  if (typeof window.conversationsData !== 'undefined') {
    const app = new RolloutViewerApp(window.conversationsData);
    setupKeyboardNavigation(app);

    // Expose for debugging
    window.rolloutViewerApp = app;
  } else {
    console.warn('RolloutViewerApp: conversationsData is not defined');
  }
}

// Support both scenarios:
// 1. DOM not loaded yet (local file, fresh page load)
// 2. DOM already loaded (wandb iframe, script injection)
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
