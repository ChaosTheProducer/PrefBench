# Mercedes-Benz E350 Core Catalog (Filtered)

## Purpose
This file is a reduced customization catalog for benchmark environment design.
It keeps only representative high-impact options and merges near-duplicate variants to control state-space growth.

## Vehicle Scope
- Brand: Mercedes-Benz
- Model: E350 Sedan (US)
- Source basis: `BENZ_gemini.md`
- Strategy: Keep high impact, merge style variants, drop low-profit accessories.
- Pricing semantics: table `msrp_delta_usd` is consumer-facing option markup; seller-side implementation cost is proxied as `0.5 * msrp_delta_usd`.

## Filtering Rules
- Keep options with strong impact on WTP, negotiation behavior, or margin.
- Merge cosmetic variants under one canonical definition.
- Drop low-price accessories with weak decision impact.
- Keep only representative options per major category.

## Canonical Dimensions

### 1) Exterior
| dimension | canonical_option | representative_options | msrp_delta_usd | keep_reason |
|---|---|---|---:|---|
| paint_color | paint_standard | Black, Polar White | 0 | Base visual baseline |
| paint_color | paint_metallic | Any metallic paint | 750 | Medium visual premium tier |
| paint_color | paint_manufaktur | Any MANUFAKTUR paint | 1750 | High-end visual premium tier |
| wheels | wheel_18_standard | 18-inch 5-spoke | 0 | Baseline wheel level |
| wheels | wheel_19_upgrade | Any 19-inch upgrade | 600 | Mid-tier wheel upgrade |
| wheels | wheel_amg_high | 20-inch/21-inch AMG wheels | 1950 | High visual/performance signal |
| exterior_style | styling_upgrade | Night Package or Illuminated Grille | 400 | Strong style/tech cue |

### 2) Interior
| dimension | canonical_option | representative_options | msrp_delta_usd | keep_reason |
|---|---|---|---:|---|
| upholstery | mb_tex | Black MB-Tex, Macchiato Beige MB-Tex, Tonka Brown MB-Tex | 0 | Interior baseline |
| upholstery | leather | Black Leather | 1620 | Clear premium upgrade |
| upholstery | nappa_leather | Nappa Leather | 2990 | Top-tier comfort/luxury signal |
| trim | standard_trim | Black wood / Piano black style | 0 | Baseline trim appearance |
| trim | premium_trim | Premium wood / metallic / star-pattern style | 150 | Premium aesthetics tier |
| comfort | multicontour_package | Multicontour Seating Package | 2950 | High-value comfort bundle |
| comfort | seat_comfort_upgrade | Ventilated front seats or Heated rear seats | 500 | Core comfort utility signal |
| comfort | soft_close_doors | Soft-close doors | 550 | Luxury convenience signal |
| audio | burmester_4d | Burmester 4D Surround Sound | 1030 | Strong premium tech signal |

### 3) Options
| dimension | canonical_option | representative_options | msrp_delta_usd | keep_reason |
|---|---|---|---:|---|
| technology | mbux_superscreen | MBUX Superscreen Package | 1500 | Core tech-adopter signal |
| safety | driver_assistance_package | Driver Assistance Package | 1950 | Core safety/automation signal |
| performance | airmatic_package | AIRMATIC Package | 3200 | High-value ride/performance signal |
| lighting | digital_light | DIGITAL LIGHT LED headlamps | 990 | Premium lighting tech signal |

## Deferred / Dropped (for state-space control)
- Exterior accessories: Body-color rear spoiler, Black rear spoiler, Chrome load-sill guard, Animated Star Logo Projectors, Mercedes-Benz Star Set, Wheel locking bolts.
- Interior small accessories: All-season floor mats, MB-Tex upper dash trim, ENERGIZING AIR Control, Black microfiber headliner.
- Low-impact options: Dashcam, First-aid kit, Rear side-impact air bags, 12.3-inch 3D instrument cluster, Winter Package.
- Overlap-heavy package options: Pinnacle Trim (deferred until explicit package constraints are introduced).

## Duplicate Handling
- `Burmester 4D Surround Sound` appears in multiple sections in source notes.
- Keep one canonical definition under `audio.burmester_4d`.
