# Changelog

## Unreleased — review fixes

Default behaviour is unchanged: with the default config, `fit_result.npz` from a synthetic
four-view run is bit-for-bit identical to the previous version (17 arrays compared).

### Fixed
- `fit`: the keypoint error was divided by a constant 534 (the panel height of the original
  reference board). It is now divided by each view's own height (`fit_terms.keypoint_term`), so
  the loss no longer depends on the panel size. On a synthetic 300 px layout the shape parameters
  moved from 0.199 away from the 534 px reference to 0.040.
- `layered`: `assert hair.is_watertight` (removed under `python -O`, and an opaque failure
  otherwise) is now a `RuntimeError` that says what is wrong.

### Added
- Config: `[subject] height_m / height_measured`, `[fit] use_view_weights / height_weight`.
  `geometry_weight` from `views.json` was parsed but never used; it now takes effect when
  `use_view_weights = true` (off by default because it changes numerical results).
- The fit report records the height prior, its weight and the view weights used.
- `fit_terms.py`, `geom2d.py`: pure, unit-tested helpers.
- Tests: config parsing, resolution independence, vectorised helpers against the original loops,
  and an opt-in end-to-end check (`HUMAN3D_SLOW_TESTS=1`) on a synthetic model
  (`tests/synthetic.py`, not a human; it only has the SMPL-X array layout).
- GitHub Actions workflow running the unit tests on CPU.

### Performance
- Scalp-cap face selection, boundary capping and the face-oval inside test were Python loops over
  every face; they are vectorised with results checked against the original loops.
