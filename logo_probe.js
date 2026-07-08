/* Lightweight, per-page logo-candidate collector, evaluated on every page
 * scrape_page.py fetches (homepage + internal links) — not just the homepage.
 *
 * brand_probe.js finds a logo from ONE page using selectors: header/nav
 * membership, alt text, filename. That fails outright on CMSs (Wix, Squarespace)
 * that don't wrap chrome in semantic <header>/<nav> tags at all.
 *
 * This script instead just harvests every plausible image/wordmark near the top
 * of whatever page it runs on, tagged with its position and identity. The Python
 * side then looks for the one candidate that recurs across multiple distinct
 * pages in roughly the same spot — the actual defining property of a logo (it's
 * the one image that's the same everywhere), independent of any markup
 * convention. A hero photo or product shot never repeats; a logo always does. */
(() => {
  const WIDGET_SELECTOR =
    '#dialogue-foundry-app, [id*="dialogue-foundry"], [class*="dialogue-foundry"], [data-dialogue-foundry-id]'
  const isOurWidget = el => Boolean(el.closest(WIDGET_SELECTOR))

  try {
    window.scrollTo(0, 0)
  } catch {
    /* not fatal */
  }

  const linksHome = el => {
    const anchor = el.closest('a[href]')
    if (!anchor) return false
    try {
      const href = new URL(anchor.href, location.href)
      return (
        href.origin === location.origin &&
        href.pathname.replace(/\/+$/, '') === ''
      )
    } catch {
      return false
    }
  }

  /* Strip resize/query params so the same logo served at different dimensions
   * on different pages (Wix's /v1/fill/w_864,h_66/... vs w_324,h_191/...) still
   * groups as one identity. */
  const normalizeUrlKey = url => {
    try {
      const parsed = new URL(url, location.href)
      return (
        'img:' +
        (parsed.origin + parsed.pathname)
          .replace(/~mv2.*$/, '')
          .toLowerCase()
      )
    } catch {
      return 'img:' + url
    }
  }

  const djb2 = text => {
    let hash = 5381
    for (let i = 0; i < text.length; i++) {
      hash = (hash * 33) ^ text.charCodeAt(i)
    }
    return (hash >>> 0).toString(36)
  }

  /* Bake computed fill/stroke onto a clone so an inline SVG (whose paint often
   * comes from `var(--token)` or an inherited class) still renders once lifted
   * out of the page's cascade. */
  const svgToDataUri = svg => {
    const clone = svg.cloneNode(true)
    const originals = [svg, ...svg.querySelectorAll('*')]
    const clones = [clone, ...clone.querySelectorAll('*')]
    for (let i = 0; i < originals.length && i < clones.length; i++) {
      const computed = getComputedStyle(originals[i])
      const props =
        originals[i].tagName.toLowerCase() === 'stop'
          ? ['stop-color']
          : ['fill', 'stroke']
      for (const prop of props) {
        const value = computed.getPropertyValue(prop)
        if (value && value !== 'none' && !value.includes('var(')) {
          clones[i].setAttribute(prop, value)
        }
      }
    }
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    const bytes = new TextEncoder().encode(clone.outerHTML)
    let binary = ''
    for (const byte of bytes) binary += String.fromCharCode(byte)
    return 'data:image/svg+xml;base64,' + btoa(binary)
  }

  const isLogoLikeSvg = svg =>
    !svg.querySelector('mask, text, foreignObject') &&
    Boolean(svg.querySelector('path, polygon, circle, rect'))

  const candidates = []
  const seenKeys = new Set()

  // Deliberately NOT scoped to header/nav — that selector assumption is exactly
  // what fails on markup-free CMS builders. Position (near the top) is the only
  // filter; recurrence across pages does the real discriminating.
  for (const img of document.querySelectorAll('img')) {
    if (isOurWidget(img)) continue
    const rect = img.getBoundingClientRect()
    if (rect.width < 16 || rect.height < 16) continue
    if (rect.top > 400) continue
    const src = img.currentSrc || img.src
    if (!src || src.startsWith('data:')) continue
    const key = normalizeUrlKey(src)
    if (seenKeys.has(key)) continue
    seenKeys.add(key)
    candidates.push({
      key,
      url: src,
      kind: 'img',
      top: Math.round(rect.top),
      left: Math.round(rect.left),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      linksHome: linksHome(img)
    })
  }

  for (const svg of document.querySelectorAll('svg')) {
    if (isOurWidget(svg)) continue
    if (svg.querySelector('svg')) continue // nested match means we grabbed a container
    const rect = svg.getBoundingClientRect()
    if (rect.width < 24 || rect.height < 12) continue
    if (rect.top > 400) continue
    if (rect.width / Math.max(rect.height, 1) < 1.2) continue
    if (!isLogoLikeSvg(svg)) continue
    const markup = svg.outerHTML
    if (markup.length > 24000) continue
    // Identity is the raw markup shape, not the baked data URI (which can shift
    // slightly if computed fill differs subtly across pages).
    const key = 'svg:' + djb2(markup.replace(/\s+/g, ''))
    if (seenKeys.has(key)) continue
    seenKeys.add(key)
    candidates.push({
      key,
      url: svgToDataUri(svg),
      kind: 'svg',
      top: Math.round(rect.top),
      left: Math.round(rect.left),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
      linksHome: linksHome(svg)
    })
  }

  return candidates.slice(0, 40)
})()
