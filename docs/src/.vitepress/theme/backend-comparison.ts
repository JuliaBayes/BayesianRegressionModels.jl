import './backend-comparison.css'

let comparisonCounter = 0

function makeButton(label: string, className: string) {
  const button = document.createElement('button')
  button.type = 'button'
  button.className = className
  button.textContent = label
  return button
}

function directChildren(root: HTMLElement, selector: string) {
  return Array.from(root.children).filter((child): child is HTMLElement =>
    child instanceof HTMLElement && child.matches(selector),
  )
}

function discoverPanes(root: HTMLElement) {
  const declared = directChildren(root, '[data-backend-pane]')
  if (declared.length >= 2) return declared

  // The concise authoring form places ordinary Markdown code fences directly
  // inside the raw comparison wrapper. VitePress renders each fence as a
  // language div; wrap those divs so the runtime can treat 2, 3, or more panes
  // uniformly without hard-coding any backend names.
  const codeBlocks = directChildren(root, 'div[class*="language-"]')
  if (codeBlocks.length < 2) return []
  return codeBlocks.map((code) => {
    const pane = document.createElement('div')
    pane.dataset.backendPane = ''
    code.before(pane)
    pane.appendChild(code)
    return pane
  })
}

function labelsFor(root: HTMLElement, panes: HTMLElement[]) {
  const declared = (root.dataset.paneLabels || '')
    .split('|')
    .map((label) => label.trim())
    .filter(Boolean)

  return panes.map((pane, index) =>
    pane.dataset.label || declared[index] || `View ${index + 1}`,
  )
}

function semanticIdsFor(root: HTMLElement, panes: HTMLElement[]) {
  const declared = (root.dataset.paneIds || '')
    .split('|')
    .map((id) => id.trim())
  const used = new Set<string>()

  return panes.map((pane, index) => {
    const requested = pane.dataset.backendPane || declared[index] || `view-${index + 1}`
    let id = requested.trim().toLowerCase().replace(/[^a-z0-9_-]+/g, '-')
    if (!id) id = `view-${index + 1}`
    if (used.has(id)) id = `${id}-${index + 1}`
    used.add(id)
    pane.dataset.backendPane = id
    return id
  })
}

function enhanceComparison(root: HTMLElement) {
  if (root.dataset.backendComparisonReady === '1') return

  const panes = discoverPanes(root)
  if (panes.length < 2) return

  root.dataset.backendComparisonReady = '1'
  const labels = labelsFor(root, panes)
  const semanticIds = semanticIdsFor(root, panes)
  const defaultOff = new Set(
    (root.dataset.defaultOff || '')
      .split('|')
      .map((id) => id.trim())
      .filter(Boolean),
  )
  const title = root.dataset.comparisonTitle || labels.join(' · ')
  const id = `backend-comparison-${++comparisonCounter}`
  let active = 0

  const toolbar = document.createElement('div')
  toolbar.className = 'backend-comparison__toolbar'

  const tablist = document.createElement('div')
  tablist.className = 'backend-comparison__tabs'
  tablist.role = 'tablist'
  tablist.setAttribute('aria-label', `Choose a view of ${title}`)

  const tabs: HTMLButtonElement[] = []
  const selectPane = (index: number, focus = false) => {
    active = index
    panes.forEach((pane, paneIndex) => {
      const selected = paneIndex === active
      pane.hidden = !selected
      tabs[paneIndex].classList.toggle('is-active', selected)
      tabs[paneIndex].setAttribute('aria-selected', String(selected))
      tabs[paneIndex].tabIndex = selected ? 0 : -1
    })
    if (focus) tabs[index].focus()
  }

  panes.forEach((pane, index) => {
    const tab = makeButton(labels[index], 'backend-comparison__tab')
    const tabId = `${id}-${index}-tab`
    const panelId = `${id}-${index}-panel`
    tab.id = tabId
    tab.role = 'tab'
    tab.setAttribute('aria-controls', panelId)
    tab.addEventListener('click', () => selectPane(index))
    tabs.push(tab)

    pane.id = panelId
    pane.classList.add('backend-comparison__panel')
    pane.role = 'tabpanel'
    pane.setAttribute('aria-labelledby', tabId)
    tablist.appendChild(tab)
  })

  tablist.addEventListener('keydown', (event) => {
    let next = active
    if (event.key === 'ArrowLeft') next = (active - 1 + panes.length) % panes.length
    else if (event.key === 'ArrowRight') next = (active + 1) % panes.length
    else if (event.key === 'Home') next = 0
    else if (event.key === 'End') next = panes.length - 1
    else return

    event.preventDefault()
    selectPane(next, true)
  })

  const expand = makeButton('Compare…', 'backend-comparison__expand')
  expand.setAttribute('aria-haspopup', 'dialog')
  toolbar.append(tablist, expand)
  root.prepend(toolbar)

  const dialog = document.createElement('dialog')
  dialog.className = 'backend-comparison__dialog'
  dialog.setAttribute('aria-label', `${title} side-by-side comparison`)

  const frame = document.createElement('div')
  frame.className = 'backend-comparison__dialog-frame'
  const header = document.createElement('header')
  header.className = 'backend-comparison__dialog-header'
  const heading = document.createElement('strong')
  heading.textContent = title
  const selector = document.createElement('fieldset')
  selector.className = 'backend-comparison__selector'
  const selectorLegend = document.createElement('legend')
  selectorLegend.textContent = 'Visible levels'
  selector.appendChild(selectorLegend)
  const close = makeButton('Close', 'backend-comparison__close')
  close.setAttribute('aria-label', 'Close side-by-side comparison')
  close.addEventListener('click', () => dialog.close())
  header.append(heading, selector, close)

  const columns = document.createElement('div')
  columns.className = 'backend-comparison__columns'
  const modalColumns: HTMLElement[] = []
  const selectors: HTMLInputElement[] = []
  panes.forEach((pane, index) => {
    const column = document.createElement('section')
    column.className = 'backend-comparison__column'
    column.dataset.backendPane = semanticIds[index]
    const columnHeading = document.createElement('h3')
    columnHeading.textContent = labels[index]
    const content = pane.cloneNode(true) as HTMLElement
    content.hidden = false
    content.removeAttribute('id')
    content.removeAttribute('role')
    content.removeAttribute('aria-labelledby')
    content.removeAttribute('data-backend-pane')
    content.classList.remove('backend-comparison__panel')
    content.querySelectorAll('.copy').forEach((button) => button.remove())
    column.append(columnHeading, content)
    columns.appendChild(column)
    modalColumns.push(column)

    const option = document.createElement('label')
    option.className = 'backend-comparison__selector-option'
    const checkbox = document.createElement('input')
    checkbox.type = 'checkbox'
    checkbox.checked = !defaultOff.has(semanticIds[index])
    checkbox.value = semanticIds[index]
    checkbox.setAttribute('aria-controls', `${id}-${semanticIds[index]}-column`)
    column.id = `${id}-${semanticIds[index]}-column`
    const optionLabel = document.createElement('span')
    optionLabel.textContent = labels[index]
    option.append(checkbox, optionLabel)
    selector.appendChild(option)
    selectors.push(checkbox)
  })

  const updateVisibleLevels = () => {
    const selected = new Set(
      selectors.filter((checkbox) => checkbox.checked).map((checkbox) => checkbox.value),
    )
    columns.style.setProperty('--backend-pane-count', String(selected.size))
    let firstVisible = true
    modalColumns.forEach((column) => {
      const visible = selected.has(column.dataset.backendPane || '')
      column.hidden = !visible
      column.classList.toggle('is-first-visible', visible && firstVisible)
      if (visible) firstVisible = false
    })
    selectors.forEach((checkbox) => {
      checkbox.disabled = checkbox.checked && selected.size <= 2
      checkbox.closest('label')?.classList.toggle('is-selected', checkbox.checked)
    })
    expand.textContent = `Compare ${selected.size} levels`
  }
  selectors.forEach((checkbox) => {
    checkbox.addEventListener('change', updateVisibleLevels)
  })
  updateVisibleLevels()

  frame.append(header, columns)
  dialog.appendChild(frame)
  root.appendChild(dialog)

  expand.addEventListener('click', () => {
    dialog.showModal()
    selectors[0].focus()
  })
  dialog.addEventListener('click', (event) => {
    if (event.target === dialog) dialog.close()
  })
  dialog.addEventListener('close', () => expand.focus())

  selectPane(0)
}

function materializeGeneratedComparisons(root: ParentNode) {
  const sentinels = Array.from(
    root.querySelectorAll<HTMLElement>('div.language-brm-comparison'),
  )
  if (root instanceof HTMLElement && root.matches('div.language-brm-comparison')) {
    sentinels.unshift(root)
  }

  sentinels.forEach((sentinel) => {
    const panes: HTMLElement[] = []
    let cursor = sentinel.nextElementSibling
    while (cursor && panes.length < 5) {
      if (!(cursor instanceof HTMLElement) || !cursor.matches('div[class*="language-"]')) {
        break
      }
      panes.push(cursor)
      cursor = cursor.nextElementSibling
    }
    if (panes.length !== 5) return

    const title = sentinel.textContent?.trim() || 'BRM model comparison'
    const comparison = document.createElement('div')
    comparison.className = 'backend-comparison'
    comparison.dataset.backendComparison = ''
    comparison.dataset.comparisonTitle = title
    comparison.dataset.paneIds = 'brm|ir|stanblocks|stan|turing'
    comparison.dataset.paneLabels =
      'BRM authoring|IR|StanBlocks model|Stan source|Turing model'
    comparison.dataset.defaultOff = 'stan'
    sentinel.replaceWith(comparison)
    panes.forEach((pane) => comparison.appendChild(pane))
  })
}

function processComparisons(root: ParentNode) {
  materializeGeneratedComparisons(root)
  const candidates = Array.from(
    root.querySelectorAll<HTMLElement>('[data-backend-comparison]'),
  )
  if (root instanceof HTMLElement && root.matches('[data-backend-comparison]')) {
    candidates.unshift(root)
  }
  candidates.forEach(enhanceComparison)
}

export function setupBackendComparisons() {
  if (typeof window === 'undefined') return

  processComparisons(document.body)
  const observer = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      for (const node of mutation.addedNodes) {
        if (node instanceof HTMLElement) processComparisons(node)
      }
    }
  })
  observer.observe(document.body, { childList: true, subtree: true })
}
