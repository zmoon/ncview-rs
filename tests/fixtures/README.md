# Synthetic fixtures

Remote-streaming tests serve byte-identical copies of these deterministic fixtures through the
in-process object-store test double. These small fixtures are intentionally committed so a clean
checkout can compile and run the tests; the ignore rules for local datasets explicitly exempt them.
Large-object coverage pads a fixture at runtime, so no cloud object or generated operational dataset
is committed. Optional live-provider checks must remain environment-gated and must never record
credentials.

Fixture builders use small, redistributable arrays with documented expected values. The
COARDS/CF NetCDF-4 fixture is generated from `coards-float32.cdl`, so tests do not depend on a
machine-specific climate-data path.

`coards-float32.nc4` models the relevant MACCity conventions: float32 `Pixel_area(date, lat, lon)`
and `MACCity(date, lat, lon)` fields, one-dimensional `date`, `lat`, and `lon` coordinates, CF
metadata, chunking, and DEFLATE compression. Their known values are 101–160 and 1–60; the
regression test reads both variables and checks their 4×5 slices. Regenerate it with:

```sh
ncgen -4 -o tests/fixtures/coards-float32.nc4 tests/fixtures/coards-float32.cdl
```

The MPAS NetCDF-3 smoke test uses a small generated field and a real mesh downloaded only by the
ignored integration test. CI downloads the UCAR `x1.2562` mesh before running that test; ordinary
`cargo test` remains deterministic and does not require network access.

The committed MPAS fixture is generated once and checked in because the current pure-Rust reader
is intentionally read-only. The fixture is covered by the normal test suite and is not rewritten by
the NetCDF-4 fixture generator.

Generated fixture hashes (SHA-256):

- `regular.nc4`: `9759f4bc44bc326c93dc0f8aeefda44f945a2673f207d7696f61c901d5fbcd48`
- `packed-fill.nc4`: `9759f4bc44bc326c93dc0f8aeefda44f945a2673f207d7696f61c901d5fbcd48`
- `curvilinear.nc4`: `abc16631330c6df1777b4b2d8b1e54e15ed211a25dbcdaffd3e1a5f178c1fbbb`
- `netcdf3-unsupported.nc`: `d71eff333e000bb7d3e1856b65e23650801ccc9c4851aa9da01eca756b49aa64`
- `corrupt.nc`: `8141db4372deed39e7986ae7cbe7faef02e34c55f73b886a01750abb5dd3442c`
- `coards-float32.nc4`: `40f0517505fd85e4e30300ee9dae9b59c71086f2ea4db1c7552c6ca87313f197`
- `mpas-small.nc`: `32ed2fc02d393a146a939d43e1ba167d7f486e12981a7e7b0fc6307dbfb88ac6`
