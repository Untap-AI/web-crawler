/* Brand-signal probe, evaluated inside the settled page by scrape_page.py.
 *
 * The old approach regexed `background-color` out of inline style="" attrs and
 * <style> blocks in the returned HTML. Modern sites keep ~all their color in
 * external stylesheets, so that saw almost nothing and the demo fell back to a
 * generic blue. Here we're already inside a real browser with the cascade
 * resolved, so we can just ask getComputedStyle.
 *
 * Returns a JSON-serializable object; the caller reads it off
 * result.js_execution_result. Everything is best-effort — a throw anywhere
 * would lose the whole scrape, so each section is independently guarded. */
(() => {
  /* crawl4ai's `simulate_user` scrolls the page to look human, so by the time we
   * run, getBoundingClientRect().top is relative to wherever it stopped. Scroll
   * back to the origin (lazy images have already loaded) so viewport coords and
   * document coords coincide and "is this in the header?" is a simple top < 200.
   * Fixed/sticky headers make the scrollY-offset alternative wrong anyway. */
  try {
    window.scrollTo(0, 0)
  } catch {
    /* not fatal */
  }

  /* Our own chat widget, if this prospect is already a customer. Its logo has
   * alt="Brand Logo" and its buttons are painted in whatever primaryColor we
   * configured last time — sampling it would make the probe read back our own
   * output instead of the site's real brand. Exclude the whole subtree. */
  const WIDGET_SELECTOR =
    '#dialogue-foundry-app, [id*="dialogue-foundry"], [class*="dialogue-foundry"], [data-dialogue-foundry-id]'
  const isOurWidget = el => Boolean(el.closest(WIDGET_SELECTOR))

  const resolveUrl = (href, base) => {
    try {
      return new URL(href, base).href
    } catch {
      return href
    }
  }

  const clamp01 = n => Math.min(1, Math.max(0, n))

  /* Parse the handful of color notations getComputedStyle actually emits
   * (rgb/rgba) plus author-written hex from CSS custom properties, which are
   * *not* resolved to rgb() the way real properties are. */
  const parseColor = value => {
    if (typeof value !== 'string') return null
    const raw = value.trim()
    if (!raw || raw === 'transparent' || raw === 'currentcolor') return null

    const rgb = raw.match(
      /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:\s*[,/]\s*([\d.%]+))?\s*\)$/i
    )
    if (rgb) {
      let alpha = 1
      if (rgb[4] !== undefined) {
        alpha = rgb[4].endsWith('%')
          ? parseFloat(rgb[4]) / 100
          : parseFloat(rgb[4])
      }
      return { r: +rgb[1], g: +rgb[2], b: +rgb[3], a: clamp01(alpha) }
    }

    const hex = raw.match(/^#([0-9a-f]{3,8})$/i)
    if (hex) {
      let h = hex[1]
      if (h.length === 3 || h.length === 4) h = [...h].map(c => c + c).join('')
      if (h.length !== 6 && h.length !== 8) return null
      return {
        r: parseInt(h.slice(0, 2), 16),
        g: parseInt(h.slice(2, 4), 16),
        b: parseInt(h.slice(4, 6), 16),
        a: h.length === 8 ? parseInt(h.slice(6, 8), 16) / 255 : 1
      }
    }
    return null
  }

  const toHex = ({ r, g, b }) =>
    '#' +
    [r, g, b]
      .map(n => Math.round(n).toString(16).padStart(2, '0'))
      .join('')
      .toLowerCase()

  const saturationOf = ({ r, g, b }) => {
    const max = Math.max(r, g, b) / 255
    const min = Math.min(r, g, b) / 255
    const l = (max + min) / 2
    if (max === min) return { s: 0, l }
    const d = max - min
    return { s: l > 0.5 ? d / (2 - max - min) : d / (max + min), l }
  }

  /* The browser's built-in unvisited/visited link colors. They show up on any
   * page with an unstyled <a>, are saturated enough to pass the filter, and are
   * never anybody's brand color. */
  const UA_DEFAULT_COLORS = new Set(['#0000ee', '#0000ff', '#551a8b'])

  /* Drop whites, blacks, greys and near-transparent overlays. These dominate
   * any usage count on a real page and are never the brand color. */
  const isBrandworthy = color => {
    if (!color || color.a < 0.6) return false
    if (UA_DEFAULT_COLORS.has(toHex(color))) return false
    const { s, l } = saturationOf(color)
    return s >= 0.15 && l >= 0.08 && l <= 0.92
  }

  /* Relaxed fallback for genuinely monochrome brands (black type on cream, no
   * saturated accent anywhere): drop the saturation requirement, but still hard
   * -exclude anything white-ish or black-ish. White/near-white is essentially
   * always a page or card background, never a chosen brand color — even for a
   * monochrome brand the "color" worth surfacing is their near-black text/logo
   * ink, not the paper it sits on. */
  const isDistinct = color => {
    if (!color || color.a < 0.6) return false
    if (UA_DEFAULT_COLORS.has(toHex(color))) return false
    const { l } = saturationOf(color)
    return l >= 0.04 && l <= 0.85
  }

  const safe = fn => {
    try {
      return fn()
    } catch {
      return null
    }
  }

  /* ---- CSS custom properties on :root ------------------------------------
   * The single strongest signal on any site built this decade: --primary,
   * --brand-color, --wp--preset--color--accent, --e-global-color-primary.
   *
   * Two ways in, because neither alone is enough:
   *   1. Enumerate authored :root rules. Free names we'd never guess, but
   *      `sheet.cssRules` throws on cross-origin stylesheets (most CDN-hosted
   *      sites), so this silently returns nothing on plenty of pages.
   *   2. Probe a fixed list of conventional names. getComputedStyle resolves a
   *      custom property no matter which stylesheet declared it, cross-origin
   *      included — so this is the one that actually fires on Stripe et al.
   * Union the two. */
  const WELL_KNOWN_VARS = [
    '--primary',
    '--primary-color',
    '--color-primary',
    '--brand',
    '--brand-color',
    '--color-brand',
    '--brand-primary',
    '--accent',
    '--accent-color',
    '--color-accent',
    '--secondary',
    '--secondary-color',
    '--color-secondary',
    '--theme-color',
    '--main-color',
    '--bs-primary', // Bootstrap 5
    '--bs-link-color',
    '--e-global-color-primary', // Elementor
    '--e-global-color-secondary',
    '--e-global-color-accent',
    '--wp--preset--color--primary', // WordPress theme.json
    '--wp--preset--color--secondary',
    '--wp--preset--color--accent',
    '--wp--preset--color--vivid-cyan-blue',
    '--global-palette1', // Kadence
    '--global-palette2',
    '--astra-global-color-0', // Astra
    '--ast-global-color-0',
    '--color-accent-1', // Shopify Dawn
    '--shopify-accent'
  ]

  const cssVariables = safe(() => {
    const names = new Set(WELL_KNOWN_VARS)

    for (const sheet of document.styleSheets) {
      let rules
      try {
        rules = sheet.cssRules
      } catch {
        continue // cross-origin stylesheet, can't introspect — that's what the list is for
      }
      if (!rules) continue
      for (const rule of rules) {
        if (!rule.style || !rule.selectorText) continue
        if (!/^(:root|html|body)\b/.test(rule.selectorText)) continue
        for (const prop of rule.style) {
          if (prop.startsWith('--')) names.add(prop)
        }
      }
    }

    const rootStyle = getComputedStyle(document.documentElement)
    const out = []
    const seen = new Set()
    for (const name of names) {
      const lower = name.toLowerCase()
      // Carousel/lightbox libraries ship their own themed defaults on :root.
      // --swiper-theme-color is #007aff on every Swiper site on earth.
      if (/swiper|slick|glide|fancybox|splide|flickity|plyr|leaflet/.test(lower))
        continue
      const color = parseColor(rootStyle.getPropertyValue(name))
      if (!color || !isBrandworthy(color)) continue
      const hex = toHex(color)
      if (seen.has(hex)) continue
      seen.add(hex)
      // Names that literally say "primary"/"brand"/"accent" outrank the rest.
      let weight = 0
      if (/primary|brand/.test(lower)) weight += 100
      if (/accent/.test(lower)) weight += 60
      if (/secondary/.test(lower)) weight += 40
      if (/theme|main|palette1|color-0|color-1|colour/.test(lower)) weight += 20
      out.push({ name, hex, weight })
    }
    return out.sort((a, b) => b.weight - a.weight).slice(0, 12)
  })

  /* ---- Computed styles on brand-bearing elements -------------------------
   * Weight each color by (usage count) x (rendered area), so a nav bar and a
   * CTA button outrank a single accented word deep in the footer. Area is
   * square-rooted to keep one giant hero section from swamping everything. */
  const collectComputed = brandworthyOnly => {
    const SELECTOR = [
      'button',
      'a',
      '[role="button"]',
      'input[type="submit"]',
      'input[type="button"]',
      'header',
      'nav',
      'h1',
      'h2',
      '.btn',
      '[class*="cta"]',
      '[class*="button"]'
    ].join(',')

    const accept = brandworthyOnly ? isBrandworthy : isDistinct

    const scores = new Map()
    const bump = (color, area, source) => {
      if (!accept(color)) return
      const hex = toHex(color)
      const entry = scores.get(hex) || { hex, score: 0, uses: 0, sources: {} }
      entry.score += Math.sqrt(Math.max(area, 1))
      entry.uses += 1
      entry.sources[source] = (entry.sources[source] || 0) + 1
      scores.set(hex, entry)
    }

    const elements = [...document.querySelectorAll(SELECTOR)].slice(0, 3000)
    for (const el of elements) {
      if (isOurWidget(el)) continue
      const rect = el.getBoundingClientRect()
      if (rect.width < 4 || rect.height < 4) continue
      const style = getComputedStyle(el)
      if (style.visibility === 'hidden' || style.display === 'none') continue

      const area = rect.width * rect.height
      const tag = el.tagName.toLowerCase()
      // A filled CTA button is the strongest brand-color signal on any page;
      // text color is weaker, and a border is weaker still.
      bump(parseColor(style.backgroundColor), area, `${tag}:bg`)
      bump(parseColor(style.color), area * 0.35, `${tag}:fg`)
      // borderTopColor resolves to `currentColor` when no border is painted,
      // which would just double-count every element's text color.
      if (parseFloat(style.borderTopWidth) > 0) {
        bump(parseColor(style.borderTopColor), area * 0.15, `${tag}:border`)
      }
    }

    return [...scores.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, 8)
      .map(({ hex, score, uses, sources }) => ({
        hex,
        score: Math.round(score),
        uses,
        sources: Object.keys(sources).slice(0, 4)
      }))
  }

  /* Brands like Allbirds are genuinely neutral — black type on cream, no
   * saturated accent anywhere. The strict filter returns nothing for them, and
   * an empty candidate list means the demo falls back to a generic blue that
   * looks nothing like the site. Retry with the saturation floor removed so we
   * still surface their near-black and their off-white. */
  const computedColors = safe(() => {
    const strict = collectComputed(true)
    return strict.length ? strict : collectComputed(false)
  })

  /* ---- <meta name="theme-color"> ---------------------------------------- */
  const themeColor = safe(() => {
    const meta = document.querySelector('meta[name="theme-color"]')
    const color = parseColor(meta?.getAttribute('content') || '')
    return color && isBrandworthy(color) ? toHex(color) : null
  })

  /* ---- Resolved fonts ----------------------------------------------------
   * The *computed* stack, not the authored declaration — tells us the webfont
   * that actually loaded rather than whatever the first rule happened to say. */
  const fonts = safe(() => {
    const clean = stack =>
      (stack || '').replace(/["']/g, '').trim() || null
    const h1 = document.querySelector('h1')
    return {
      body: clean(getComputedStyle(document.body).fontFamily),
      heading: h1 ? clean(getComputedStyle(h1).fontFamily) : null
    }
  })

  /* ---- Logo --------------------------------------------------------------
   * A logo is a small, wide-ish image pinned to the top-left of the header. It
   * is NOT the 1200x600 hero background that happens to be the biggest <img> on
   * the page — scoring by raw area (as the old heuristic did) picks the hero
   * almost every time. Score by header membership and logo-ish geometry, and
   * treat area as a *penalty* past a plausible logo size.
   *
   * `currentSrc` resolves srcset and lazy-loaded images, which reading `src`
   * misses. Inline <svg> logos (Stripe, Vercel, most modern sites) have no URL
   * at all, so serialize them to a data: URI — the widget renders that fine. */
  /* Which signal family each candidate kind comes from. Publisher-declared
   * images ('structured') are collected and quota'd separately from things we
   * inferred by walking the DOM, because the two can't be ranked against each
   * other by score — see the quota at the end of the logos block. */
  const SOURCE_CLASS = {
    'ld+json': 'structured',
    og: 'structured',
    twitter: 'structured',
    tile: 'structured',
    manifest: 'structured',
    'apple-touch-icon': 'structured',
    favicon: 'structured',
    img: 'dom',
    svg: 'dom'
  }

  const HEADER_PARTS = [
    'header',
    'nav',
    '[class*="header"]',
    '[id*="header"]',
    '[class*="navbar"]',
    '[role="banner"]'
  ]
  const HEADER_SELECTOR = HEADER_PARTS.join(', ')
  // Soft penalty threshold for the TEXT-fallback ranking only (used when no
  // screenshot is available to run vision against) — not a hard cutoff. A
  // legitimate hero-style emblem (common for wineries/farms/boutique brands)
  // can easily render at 350-450px; gating candidates out at this size is what
  // caused the real West Hills Vineyards logo to never even reach the vision
  // step. Actual full-bleed hero PHOTOS are rejected separately, by HERO_AREA.
  const MAX_LOGO_AREA = 60000
  // Nothing genuinely logo-shaped is bigger than this — past it, it's
  // unambiguously a hero/background photo. Hard exclusion.
  const HERO_AREA_CUTOFF = 500000 // ~700x700, or a 1000x500 hero banner
  const MAX_SVG_BYTES = 24000

  const encodeSvg = markup => {
    const bytes = new TextEncoder().encode(markup)
    let binary = ''
    for (const byte of bytes) binary += String.fromCharCode(byte)
    return 'data:image/svg+xml;base64,' + btoa(binary)
  }

  /* Serializing an inline <svg> straight out of the DOM produces markup whose
   * paint comes from the page's cascade — `fill="var(--hds-color-text-solid)"`
   * (Stripe) or a fill inherited from a stylesheet. Inside a data: URI neither
   * resolves, and the logo renders invisible. Bake the *computed* fill/stroke
   * onto every node before serializing. */
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
      const width = computed.getPropertyValue('stroke-width')
      if (width && width !== '1px') clones[i].setAttribute('stroke-width', width)
    }
    clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg')
    return encodeSvg(clone.outerHTML)
  }

  /* Decorative or functional SVGs that look logo-shaped: sign-in cutout masks,
   * animated text, icon sprites. A real mark is drawn from paths. */
  const isLogoLikeSvg = svg =>
    !svg.querySelector('mask, text, foreignObject') &&
    Boolean(svg.querySelector('path, polygon, circle, rect'))

  const svgNameScore = svg => {
    const hay = [
      svg.getAttribute('aria-label'),
      svg.getAttribute('class'),
      svg.getAttribute('id'),
      svg.querySelector('title')?.textContent,
      svg.querySelector('[id]')?.id
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase()
    if (/logo|brand|wordmark/.test(hay)) return 80
    if (/menu|hamburger|search|cart|close|arrow|chevron|social/.test(hay)) return -120
    return 0
  }

  /* A logo is almost always wrapped in the anchor that points home. */
  const linksHome = el => {
    const anchor = el.closest('a[href]')
    if (!anchor) return false
    try {
      const href = new URL(anchor.href, location.href)
      return href.origin === location.origin && href.pathname.replace(/\/+$/, '') === ''
    } catch {
      return false
    }
  }

  /* The wrapping anchor's accessible name often carries the brand even when the
   * image itself is anonymous — apple.com's mark is an unlabeled <svg> inside
   * <a aria-label="Apple">. Surfaced on each candidate so the vision selection
   * step can show it as context, not used for scoring here. */
  const anchorLabelOf = el => {
    const anchor = el.closest('a[href]')
    if (!anchor) return null
    const label = (
      anchor.getAttribute('aria-label') ||
      anchor.getAttribute('title') ||
      ''
    ).trim()
    return label ? label.slice(0, 80) : null
  }

  /* Authoritative when present: schema.org Organization.logo. Wix, Shopify and
   * most CMSs emit it, and it survives hashed filenames that name-scoring can't
   * read. `image` is only taken off an Organization-ish node — on a Product or
   * Article node it's a product shot or article hero, not the brand's logo. */
  const ORGANIZATION_TYPES =
    /^(Organization|Corporation|LocalBusiness|OnlineBusiness|NewsMediaOrganization|EducationalOrganization|GovernmentOrganization|NGO|PerformingGroup|SportsOrganization|Restaurant|Store|Brand|WebSite)$/i
  const isOrganizationNode = node => {
    const type = node['@type']
    const types = Array.isArray(type) ? type : [type]
    return types.some(t => typeof t === 'string' && ORGANIZATION_TYPES.test(t))
  }

  const structuredLogos = () => {
    const found = []
    const visit = node => {
      if (!node || typeof node !== 'object') return
      if (Array.isArray(node)) return node.forEach(visit)
      const organization = isOrganizationNode(node)
      const keys = organization ? ['logo', 'logoUrl', 'image'] : ['logo', 'logoUrl']
      for (const key of keys) {
        const value = node[key]
        const url = typeof value === 'string' ? value : value?.url
        if (typeof url === 'string' && /^https?:/.test(url)) {
          found.push({ url, weight: key === 'image' ? 30 : 120 })
        }
      }
      Object.values(node).forEach(visit)
    }
    for (const script of document.querySelectorAll(
      'script[type="application/ld+json"]'
    )) {
      try {
        visit(JSON.parse(script.textContent))
      } catch {
        /* a malformed blob shouldn't sink the probe */
      }
    }
    return found
  }

  const logos = safe(() => {
    /* Score over everything that might name the image, not just the URL — Wix
     * and friends serve `c08e45_c5e66d...~mv2.png` but set alt="Brand Logo". */
    const nameScore = (...parts) => {
      const lower = parts.filter(Boolean).join(' ').toLowerCase()
      let n = 0
      if (lower.includes('logo')) n += 60
      if (lower.includes('brand')) n += 30
      if (lower.includes('icon')) n += 10
      if (/sprite|placeholder|hero|banner|background|bg-/.test(lower)) n -= 60
      return n
    }

    /* Logos come in three common shapes: wide wordmarks (1:1 to 8:1), round
     * seals/emblems (roughly square), and narrow pictorial glyphs (Apple's
     * 14x44 mark). Reward all of them; punish anything approaching hero
     * dimensions regardless of shape. This score only pre-ranks which
     * candidates ride along to the vision selection step — it decides nothing
     * on its own. */
    const geometryScore = rect => {
      const area = rect.width * rect.height
      const ratio = rect.width / Math.max(rect.height, 1)
      let n = 0
      if (ratio >= 1 && ratio <= 8) n += 30 // wordmark proportions
      else if (ratio >= 0.6 && ratio < 1) n += 25 // seal/emblem proportions
      else if (ratio >= 0.2 && ratio < 0.6) n += 15 // narrow glyph mark
      if (rect.height <= 140) n += 25
      if (area > MAX_LOGO_AREA) n -= 50 + Math.min(100, area / 20000)
      return n
    }

    const rectOf = rect => ({
      top: Math.round(rect.top),
      left: Math.round(rect.left),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    })

    const candidates = []

    // NOT scoped to header/nav and not cut off after the first screenful — a
    // centered hero emblem (common for wineries, farms, boutique brands) can
    // sit well below the fold on a tall homepage. Position is recorded as
    // context, not used as a hard filter here, since restricting it is
    // exactly what causes misses like this.
    // A 12px floor, not 16 — small-but-real marks exist, and anything smaller
    // genuinely can't be a displayable logo.
    for (const img of [...document.querySelectorAll('img')].slice(0, 400)) {
      if (isOurWidget(img)) continue
      const rect = img.getBoundingClientRect()
      if (rect.width < 12 || rect.height < 12) continue
      const src = img.currentSrc || img.src
      if (!src || src.startsWith('data:')) continue
      const inHeader = Boolean(img.closest(HEADER_SELECTOR))
      const named = nameScore(
        src,
        img.alt,
        img.title,
        img.className,
        img.id,
        img.parentElement?.className
      )
      const area = rect.width * rect.height
      // Deliberately lenient: this pool feeds vision selection, which does the
      // real discriminating by *looking at the picture*. Only hard-exclude
      // what's unambiguous — an unnamed, non-header image below a generous
      // hero-region cutoff, or anything at true hero-photo scale regardless of
      // position. Everything else rides through for vision to judge; `score`
      // only pre-ranks which candidates make the pool cap.
      if (area > HERO_AREA_CUTOFF) continue
      if (!inHeader && named <= 0 && rect.top > 900) continue
      const homeLink = linksHome(img)
      candidates.push({
        url: src,
        kind: 'img',
        rect: rectOf(rect),
        area: Math.round(area),
        alt: (img.alt || '').trim().slice(0, 80) || null,
        anchorLabel: anchorLabelOf(img),
        linksHome: homeLink,
        score: Math.round(
          (inHeader ? 80 : 0) +
            (homeLink ? 70 : 0) +
            named +
            geometryScore(rect) -
            rect.top / 80 -
            rect.left / 40
        )
      })
    }

    // Same 12px floor as imgs. The old 24px width floor silently dropped
    // apple.com's real mark — a 14x44 <svg> in the nav — before selection ever
    // saw it, which is how a nav-label glyph got shipped as the logo.
    for (const svg of [...document.querySelectorAll('svg')].slice(0, 60)) {
      if (isOurWidget(svg)) continue
      if (svg.querySelector('svg')) continue // nested match means we grabbed a container
      const rect = svg.getBoundingClientRect()
      if (rect.width < 12 || rect.height < 12) continue
      if (!isLogoLikeSvg(svg)) continue
      const inHeader = Boolean(svg.closest(HEADER_SELECTOR))
      const named = svgNameScore(svg)
      const shapeScore = geometryScore(rect)
      // An inline <svg> is never a hero photo (those are always raster), so the
      // hero-scale exclusion img candidates need doesn't apply here — just keep
      // it near the top of the page unless it's clearly named/shaped as a mark.
      if (!inHeader && named <= 0 && shapeScore <= 0 && rect.top > 900) continue
      if (svg.outerHTML.length > MAX_SVG_BYTES) continue
      const url = svgToDataUri(svg)
      if (url.length > MAX_SVG_BYTES * 2) continue
      const homeLink = linksHome(svg)
      const label = (
        svg.getAttribute('aria-label') ||
        svg.querySelector('title')?.textContent ||
        ''
      ).trim()
      candidates.push({
        url,
        kind: 'svg',
        rect: rectOf(rect),
        area: Math.round(rect.width * rect.height),
        alt: label.slice(0, 80) || null,
        anchorLabel: anchorLabelOf(svg),
        linksHome: homeLink,
        score: Math.round(
          (inHeader ? 90 : 40) +
            named +
            (homeLink ? 70 : 0) +
            shapeScore -
            rect.top / 80 -
            rect.left / 40
        )
      })
    }

    for (const { url, weight } of structuredLogos()) {
      candidates.push({ url, kind: 'ld+json', area: 0, score: weight })
    }

    /* Publisher-declared images. og:image is frequently a hero/marketing shot
     * rather than a mark, so it cannot be trusted outright — but it's often the
     * *only* clean signal on sites whose header logo is an inline SVG glyph
     * (apple.com serves its real mark here as open_graph_logo.png). Emit it as
     * a candidate and let the vision step decide by looking at it. */
    const ogImage = document
      .querySelector('meta[property="og:image"], meta[name="og:image"]')
      ?.getAttribute('content')
    if (ogImage) {
      candidates.push({ url: resolveUrl(ogImage, location.href), kind: 'og', area: 0, score: 5 })
    }
    const twitterImage = document
      .querySelector('meta[name="twitter:image"], meta[property="twitter:image"]')
      ?.getAttribute('content')
    if (twitterImage) {
      candidates.push({ url: resolveUrl(twitterImage, location.href), kind: 'twitter', area: 0, score: 4 })
    }

    /* Windows tile image — brand-owned, usually the mark on a solid tile. */
    const tileImage = document
      .querySelector('meta[name="msapplication-TileImage"]')
      ?.getAttribute('content')
    if (tileImage) {
      candidates.push({ url: resolveUrl(tileImage, location.href), kind: 'tile', area: 0, score: 7 })
    }

    /* Always present, rarely the best *display* logo (too low-res), but a
     * strong signal for color specifically — favicons are commonly simplified
     * to a single dominant brand chip precisely because they render at 16-32px.
     * kind is tagged distinctly so a caller can prioritize these for color
     * sampling even when they'd never win as the displayed logoUrl. */
    const iconHref = document
      .querySelector('link[rel="apple-touch-icon"], link[rel="apple-touch-icon-precomposed"]')
      ?.getAttribute('href')
    if (iconHref) {
      candidates.push({ url: resolveUrl(iconHref, location.href), kind: 'apple-touch-icon', area: 0, score: 8 })
    }
    const faviconHref = document
      .querySelector('link[rel="icon"], link[rel="shortcut icon"]')
      ?.getAttribute('href')
    if (faviconHref) {
      candidates.push({ url: resolveUrl(faviconHref, location.href), kind: 'favicon', area: 0, score: 6 })
    }

    const seen = new Set()
    const deduped = candidates
      .sort((a, b) => b.score - a.score)
      .filter(c => !seen.has(c.url) && seen.add(c.url))
      .map(c => ({ ...c, sourceClass: SOURCE_CLASS[c.kind] || 'dom' }))

    /* Quota by source class rather than taking the global top-N by score.
     * Scores from different signal families aren't commensurable — a
     * publisher-declared logo scores 120 while a header <svg> scores 320, and
     * neither number means the same thing. Score-sorting therefore lets one
     * noisy family evict every other, which is exactly how apple.com's real
     * mark got pushed out by nav-glyph candidates. Guarantee slots per family
     * instead; the vision step downstream does the actual discriminating by
     * looking at the images. */
    const take = (sourceClass, limit) =>
      deduped.filter(c => c.sourceClass === sourceClass).slice(0, limit)
    return [...take('structured', 5), ...take('dom', 7)]
  })

  const title = safe(() => {
    const siteName = document
      .querySelector('meta[property="og:site_name"]')
      ?.getAttribute('content')
    return (siteName || document.title || '').trim().slice(0, 120) || null
  })

  /* The manifest's icons are brand-owned and high-res, but reading them means
   * fetching and parsing JSON — this probe is synchronous, so hand the URL to
   * the Python side and let it resolve the icons into extra candidates. */
  const manifestHref = safe(() => {
    const href = document
      .querySelector('link[rel="manifest"]')
      ?.getAttribute('href')
    return href ? resolveUrl(href, location.href) : null
  })

  return {
    cssVariables: cssVariables || [],
    computedColors: computedColors || [],
    themeColor: themeColor || null,
    fonts: fonts || { body: null, heading: null },
    logos: logos || [],
    manifestHref: manifestHref || null,
    title: title || null
  }
})()
